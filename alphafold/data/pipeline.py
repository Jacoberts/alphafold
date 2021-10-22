# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Mapping, Optional, Sequence, List, Dict
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
#from alphafold.data.tools import mmseqs as mmseqs2
import numpy as np
from dataclasses import dataclass
from string import ascii_uppercase
import pickle

from pathlib import Path
import random


# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


@dataclass
class Mock_Template_Result():
    features: FeatureDict


def make_mock_template(query_sequence: str):
    # since alphafold's model requires a template input
    # we create a blank example w/ zero input, confidence -1
    ln = len(query_sequence)
    output_templates_sequence = "-" * ln
    output_confidence_scores = np.full(ln, -1)
    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3))
    templates_all_atom_masks = np.zeros(
        (ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence,
        templates.residue_constants.HHBLITS_AA_TO_ID)
    template_features = {
        'template_all_atom_positions': templates_all_atom_positions[None],
        'template_all_atom_masks': templates_all_atom_masks[None],
        'template_sequence': np.array([f'none'.encode()]),
        'template_aatype': np.array(templates_aatype)[None],
        'template_confidence_scores': output_confidence_scores[None],
        'template_domain_names': np.array([f'none'.encode()]),
        'template_release_date': np.array([f'none'.encode()])
    }
    return Mock_Template_Result(features=template_features)


def _placeholder_template_feats(num_templates_, num_res_):
    return {
        'template_aatype':
        np.zeros([num_templates_, num_res_, 22], np.float32),
        'template_all_atom_masks':
        np.zeros([num_templates_, num_res_, 37, 3], np.float32),
        'template_all_atom_positions':
        np.zeros([num_templates_, num_res_, 37], np.float32),
        'template_domain_names':
        np.zeros([num_templates_], np.float32),
        'template_sum_probs':
        np.zeros([num_templates_], np.float32),
    }


def make_sequence_features(sequence: str,
                           description: str,
                           num_res: int,
                           homooligomer: int = 1) -> FeatureDict:
    """Constructs a feature dict of sequence features."""
    Ln: int = len(sequence)
    Ls: Sequence[int] = [Ln] * homooligomer
    num_res = num_res * homooligomer
    sequence = sequence * homooligomer

    features = {}
    features['aatype'] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True)
    features['between_segment_residues'] = np.zeros((num_res, ),
                                                    dtype=np.int32)
    features['domain_name'] = np.array([description.encode('utf-8')],
                                       dtype=np.object_)
    features['residue_index'] = np.array(range(num_res), dtype=np.int32)
    features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
    features['sequence'] = np.array([sequence.encode('utf-8')],
                                    dtype=np.object_)

    if homooligomer > 1:
        # add big enough number to residue index to indicate chain breaks between oligomers
        idx_res = features['residue_index']
        L_prev = 0
        # Ls: number of residues in each chain
        for L_i in Ls[:-1]:
            idx_res[L_prev + L_i:] += 200
            L_prev += L_i
        chains = list("".join(
            [ascii_uppercase[n] * L for n, L in enumerate(Ls)]))
        features['residue_index'] = idx_res
    return features


def make_msa_features(msas: Sequence[Sequence[str]],
                      deletion_matrices: Sequence[parsers.DeletionMatrix],
                      Ln: int,
                      msa_size_gb: float,
                      homooligomer: int = 1) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    # Flatten and denormalize the MSA. The denormalized form has every
    # sequence from all the MSAs, times the number of homooligomers.
    denorm_msa = []
    denorm_deletion_matrix = []
    for msa_idx, (msa, deletion_matrix) in enumerate(zip(msas, deletion_matrices)):
        if not msa:
            raise ValueError(
                f'MSA {msa_idx} must contain at least one sequence.')
        for sequence, deletion_row in zip(msa, deletion_matrix):
            for olig_idx in range(homooligomer):
                L = Ln * olig_idx
                R = Ln * (homooligomer - (olig_idx + 1))
                denorm_msa.append("-" * L + sequence + "-" * R)
                denorm_deletion_matrix.append([0] * L + deletion_row + [0] * R)

    # 1.99 GB Max size, size of row in msa array = Ln * 4 bytes (int32)
    max_msa_sequences = (msa_size_gb * 1024 * 1024 * 1024) // (Ln * homooligomer * 4)
 
    # Randomly select a subset of the flattened form and convert to ints.

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for index in random.sample(range(len(denorm_msa)), k=len(denorm_msa)):
        sequence = denorm_msa[index]
        deletion_row = denorm_deletion_matrix[index]

        # Don't add duplicate sequences to the MSA.
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)

        int_msa.append(
            [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
        deletion_matrix.append(deletion_row)

        if len(seen_sequences) >= max_msa_sequences:
            break

    num_res = len(denorm_msa[0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array([num_alignments] * num_res,
                                          dtype=np.int32)
    return features


def homooligomerize(msas, deletion_matrices, homooligomer=1):
    '''
  From https://github.com/sokrypton/ColabFold/blob/main/beta/colabfold.py
  '''
    if homooligomer == 1:
        return msas, deletion_matrices
    else:
        new_msas = []
        new_mtxs = []
        for o in range(homooligomer):
            for msa, mtx in zip(msas, deletion_matrices):
                num_res = len(msa[0])
                L = num_res * o
                R = num_res * (homooligomer - (o + 1))
                new_msas.append(["-" * L + s + "-" * R for s in msa])
                new_mtxs.append([[0] * L + m + [0] * R for m in mtx])
    return new_msas, new_mtxs


def homooligomerize_heterooligomer(msas, deletion_matrices, lengths,
                                   homooligomers):
    '''
  From https://github.com/sokrypton/ColabFold/blob/main/beta/colabfold.py

  ----- inputs -----
  msas: list of msas
  deletion_matrices: list of deletion matrices
  lengths: list of lengths for each component in complex
  homooligomers: list of number of homooligomeric copies for each component
  ----- outputs -----
  (msas, deletion_matrices)
  '''
    if max(homooligomers) == 1:
        return msas, deletion_matrices

    elif len(homooligomers) == 1:
        return homooligomerize(msas, deletion_matrices, homooligomers[0])

    else:
        frag_ij = [[0, lengths[0]]]
        for length in lengths[1:]:
            j = frag_ij[-1][-1]
            frag_ij.append([j, j + length])

        # for every msa
        mod_msas, mod_mtxs = [], []
        for msa, mtx in zip(msas, deletion_matrices):
            mod_msa, mod_mtx = [], []
            # for every sequence
            for n, (s, m) in enumerate(zip(msa, mtx)):
                # split sequence
                _s, _m, _ok = [], [], []
                for i, j in frag_ij:
                    _s.append(s[i:j])
                    _m.append(m[i:j])
                    _ok.append(max([o != "-" for o in _s[-1]]))

                if n == 0:
                    # if first query sequence
                    mod_msa.append("".join(
                        [x * h for x, h in zip(_s, homooligomers)]))
                    mod_mtx.append(
                        sum([x * h for x, h in zip(_m, homooligomers)], []))

                elif sum(_ok) == 1:
                    # elif one fragment: copy each fragment to every homooligomeric copy
                    a = _ok.index(True)
                    for h_a in range(homooligomers[a]):
                        _blank_seq = [["-" * l] * h
                                      for l, h in zip(lengths, homooligomers)]
                        _blank_mtx = [[[0] * l] * h
                                      for l, h in zip(lengths, homooligomers)]
                        _blank_seq[a][h_a] = _s[a]
                        _blank_mtx[a][h_a] = _m[a]
                        mod_msa.append("".join(
                            ["".join(x) for x in _blank_seq]))
                        mod_mtx.append(
                            sum([sum(x, []) for x in _blank_mtx], []))
                else:
                    # else: copy fragment pair to every homooligomeric copy pair
                    for a in range(len(lengths) - 1):
                        if _ok[a]:
                            for b in range(a + 1, len(lengths)):
                                if _ok[b]:
                                    for h_a in range(homooligomers[a]):
                                        for h_b in range(homooligomers[b]):
                                            _blank_seq = [
                                                ["-" * l] * h for l, h in zip(
                                                    lengths, homooligomers)
                                            ]
                                            _blank_mtx = [
                                                [[0] * l] * h for l, h in zip(
                                                    lengths, homooligomers)
                                            ]
                                            for c, h_c in zip([a, b],
                                                              [h_a, h_b]):
                                                _blank_seq[c][h_c] = _s[c]
                                                _blank_mtx[c][h_c] = _m[c]
                                            mod_msa.append("".join([
                                                "".join(x) for x in _blank_seq
                                            ]))
                                            mod_mtx.append(
                                                sum([
                                                    sum(x, [])
                                                    for x in _blank_mtx
                                                ], []))
            mod_msas.append(mod_msa)
            mod_mtxs.append(mod_mtx)
        return mod_msas, mod_mtxs


def chain_break(idx_res, Ls, length=200):
    '''From https://github.com/sokrypton/ColabFold/blob/main/beta/colabfold.py'''
    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i:] += length
        L_prev += L_i
    return idx_res


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""
    def __init__(self,
                 jackhmmer_binary_path: str,
                 hhblits_binary_path: str,
                 hhsearch_binary_path: str,
                 uniref90_database_path: str,
                 mgnify_database_path: str,
                 bfd_database_path: Optional[str],
                 uniclust30_database_path: Optional[str],
                 small_bfd_database_path: Optional[str],
                 pdb70_database_path: str,
                 template_featurizer: templates.TemplateHitFeaturizer,
                 mmseqs_binary_path: str,
                 mmseqs_uniref50_database_path: str,
                 mmseqs_mgnify_database_path: str,
                 mmseqs_small_bfd_database_path: str,
                 mmseqs: bool,
                 use_small_bfd: bool,
                 tmp_dir: Path,
                 mgnify_max_hits: int = 501,
                 uniref_max_hits: int = 25000,
                 bfd_max_hits: int = 25000):
        """Constructs a feature dict for a given FASTA file."""
        self._use_small_bfd = use_small_bfd
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path,
            tmp_dir=tmp_dir,
            get_tblout=True)
        if use_small_bfd:
            self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=small_bfd_database_path,
                tmp_dir=tmp_dir,
                get_tblout=True)
        else:
            self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
                binary_path=hhblits_binary_path,
                databases=[bfd_database_path, uniclust30_database_path],
                tmp_dir=tmp_dir)
        self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=mgnify_database_path,
            tmp_dir=tmp_dir,
            get_tblout=True)
        self.hhsearch_pdb70_runner = hhsearch.HHSearch(
            binary_path=hhsearch_binary_path,
            databases=[pdb70_database_path],
            tmp_dir=tmp_dir)
        #self.mmseqs_runner = mmseqs2.MMSeqs(
        #    binary_path=mmseqs_binary_path,
        #    uniref50_database_path=mmseqs_uniref50_database_path,
        #    mgnify_database_path=mmseqs_mgnify_database_path,
        #    small_bfd_database_path=mmseqs_small_bfd_database_path)
        self.template_featurizer = template_featurizer
        self.mgnify_max_hits = mgnify_max_hits
        self.uniref_max_hits = uniref_max_hits
        self.bfd_max_hits = bfd_max_hits

    def process(self,
                input_fasta_path: str,
                msa_output_dir: str,
                msa_size_gb: float,
                homooligomer: str = '1') -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        assert len(homooligomer) == 1
        homooligomer = int(homooligomer)
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        if not os.path.exists(uniref90_out_path):
            jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
                input_fasta_path)[0]
            with open(uniref90_out_path, 'w') as f:
                f.write(jackhmmer_uniref90_result['sto'])
        else:
            jackhmmer_uniref90_result = {}
            with open(uniref90_out_path, 'r') as f:
                jackhmmer_uniref90_result['sto'] = f.read()

        mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
        if not os.path.exists(mgnify_out_path):
            jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
                input_fasta_path)[0]
            with open(mgnify_out_path, 'w') as f:
                f.write(jackhmmer_mgnify_result['sto'])
        else:
            jackhmmer_mgnify_result = {}
            with open(uniref90_out_path, 'r') as f:
                jackhmmer_mgnify_result['sto'] = f.read()

        pdb70_out_path = os.path.join(msa_output_dir, 'pdb70_hits.hhr')
        if not os.path.exists(pdb70_out_path):
            uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
                jackhmmer_uniref90_result['sto'],
                max_sequences=self.uniref_max_hits)
            hhsearch_result = self.hhsearch_pdb70_runner.query(
                uniref90_msa_as_a3m)
            with open(pdb70_out_path, 'w') as f:
                f.write(hhsearch_result)
        else:
            with open(pdb70_out_path, 'r') as f:
                hhsearch_result = f.read()

        uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
            jackhmmer_uniref90_result['sto'])
        mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
            jackhmmer_mgnify_result['sto'])
        hhsearch_hits = parsers.parse_hhr(hhsearch_result)
        #mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
        #mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]
        #uniref90_msa = uniref90_msa[:self.uniref_max_hits]
        #uniref90_deletion_matrix = uniref90_deletion_matrix[:self.uniref_max_hits]

        if self._use_small_bfd:
            bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
            if not os.path.exists(bfd_out_path):
                jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
                    input_fasta_path)[0]
                with open(bfd_out_path, 'w') as f:
                    f.write(jackhmmer_small_bfd_result['sto'])
            else:
                jackhmmer_small_bfd_result = {}
                with open(bfd_out_path, 'r') as f:
                    jackhmmer_small_bfd_result['sto'] = f.read()

            bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
                jackhmmer_small_bfd_result['sto'])
        else:
            bfd_out_path = os.path.join(msa_output_dir,
                                        'bfd_uniclust_hits.a3m')
            if not os.path.exists(bfd_out_path):
                hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
                    input_fasta_path)
                with open(bfd_out_path, 'w') as f:
                    f.write(hhblits_bfd_uniclust_result['a3m'])
            else:
                hhblits_bfd_uniclust_result = {}
                with open(bfd_out_path, 'r') as f:
                    hhblits_bfd_uniclust_result['a3m'] = f.read()

            bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
                hhblits_bfd_uniclust_result['a3m'])
        #bfd_msa = bfd_msa[:self.bfd_max_hits]
        #bfd_deletion_matrix = bfd_deletion_matrix[:self.bfd_max_hits]

        if homooligomer > 1:
            templates_result = make_mock_template(
                query_sequence=input_sequence * homooligomer)
        else:
            templates_result = self.template_featurizer.get_templates(
                query_sequence=input_sequence,
                query_pdb_code=None,
                query_release_date=None,
                hits=hhsearch_hits)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
            homooligomer=homooligomer)

        msa_features = make_msa_features(
            msas=(uniref90_msa, bfd_msa, mgnify_msa),
            deletion_matrices=(uniref90_deletion_matrix, bfd_deletion_matrix,
                               mgnify_deletion_matrix),
            Ln=len(input_sequence),
            msa_size_gb=msa_size_gb,
            homooligomer=homooligomer)

        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
        logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
        logging.info('Final (deduplicated) MSA size: %d sequences.',
                     msa_features['num_alignments'][0])
        logging.info(
            'Total number of templates (NB: this can include bad '
            'templates and is later filtered to top 4): %d.',
            templates_result.features['template_domain_names'].shape[0])

        return {
            **sequence_features,
            **msa_features,
            **templates_result.features
        }

    def create_msas(self, input_fasta_path: str, msa_output_dir: str) -> None:
        """Runs alignment tools on the input sequence."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)
        prefix: str = input_description
        dbs = []

        pickled_msa_path = os.path.join(msa_output_dir, f"{prefix}.pickle")
        logging.info(f'Pickled MSA Path: {pickled_msa_path}')
        if not os.path.exists(pickled_msa_path):
            logging.info(
                f'Pickled MSA Path does not exist yet: {pickled_msa_path}')
            uniref90_out_path = os.path.join(msa_output_dir,
                                             f'{prefix}_uniref90_hits.sto')
            if not os.path.exists(uniref90_out_path):
                jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
                    input_fasta_path)[0]
                with open(uniref90_out_path, 'w') as f:
                    f.write(jackhmmer_uniref90_result['sto'])
            else:
                jackhmmer_uniref90_result = {}
                with open(uniref90_out_path, 'r') as f:
                    jackhmmer_uniref90_result['sto'] = f.read()

            mgnify_out_path = os.path.join(msa_output_dir,
                                           f'{prefix}_mgnify_hits.sto')
            if not os.path.exists(mgnify_out_path):
                jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
                    input_fasta_path)[0]
                with open(mgnify_out_path, 'w') as f:
                    f.write(jackhmmer_mgnify_result['sto'])
            else:
                jackhmmer_mgnify_result = {}
                with open(uniref90_out_path, 'r') as f:
                    jackhmmer_mgnify_result['sto'] = f.read()

            if self._use_small_bfd:
                bfd_out_path = os.path.join(msa_output_dir,
                                            f'{prefix}_small_bfd_hits.a3m')
                if not os.path.exists(bfd_out_path):
                    jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
                        input_fasta_path)[0]
                    with open(bfd_out_path, 'w') as f:
                        f.write(jackhmmer_small_bfd_result['sto'])
                else:
                    jackhmmer_small_bfd_result = {}
                    with open(bfd_out_path, 'r') as f:
                        jackhmmer_small_bfd_result['sto'] = f.read()
                dbs.append(('smallbfd', jackhmmer_small_bfd_result))
            else:
                bfd_out_path = os.path.join(msa_output_dir,
                                            f'{prefix}_bfd_uniclust_hits.a3m')
                if not os.path.exists(bfd_out_path):
                    hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
                        input_fasta_path)
                    with open(bfd_out_path, 'w') as f:
                        f.write(hhblits_bfd_uniclust_result['a3m'])
                else:
                    hhblits_bfd_uniclust_result = {}
                    with open(bfd_out_path, 'r') as f:
                        hhblits_bfd_uniclust_result['a3m'] = f.read()
                dbs.append(('bfd', hhblits_bfd_uniclust_result))

            dbs.append(('uniref90', jackhmmer_uniref90_result))
            dbs.append(('mgnify', jackhmmer_mgnify_result))

            msas = []
            deletion_matrices = []
            names = []
            for db_name, db_results in dbs:
                try:
                    unsorted_results = []
                    for i, result in enumerate(db_results):
                        if db_name == 'bfd':
                            msa, deletion_matrix = parsers.parse_a3m(
                                db_results['a3m'])
                        else:
                            msa, deletion_matrix, target_names = parsers.parse_stockholm(
                                db_results['sto'])
                        e_values_dict = parsers.parse_e_values_from_tblout(
                            db_results['tbl'])
                        e_values = [
                            e_values_dict[t.split('/')[0]]
                            for t in target_names
                        ]
                        zipped_results = zip(msa, deletion_matrix,
                                             target_names, e_values)
                        if i != 0:
                            # Only take query from the first chunk
                            zipped_results = [
                                x for x in zipped_results if x[2] != 'query'
                            ]
                        unsorted_results.extend(zipped_results)
                    sorted_by_evalue = sorted(unsorted_results,
                                              key=lambda x: x[3])
                    db_msas, db_deletion_matrices, db_names, _ = zip(
                        *sorted_by_evalue)
                    if db_msas:
                        msas.append(db_msas)
                        deletion_matrices.append(db_deletion_matrices)
                        names.append(db_names)
                        msa_size = len(set(db_msas))
                        logging.info(
                            f'{msa_size} Sequences Found in {db_name}')
                except:
                    unsorted_results = []
                    msa, deletion_matrix, target_names = parsers.parse_stockholm(
                        db_results['sto'])
                    zipped_results = zip(msa, deletion_matrix, target_names)
                    if db_name != 'smallbfd':
                        zipped_results = [
                            x for x in zipped_results
                            if x[2] != input_description
                        ]
                    unsorted_results.extend(zipped_results)
                    if not len(unsorted_results):
                        continue  # if no hits found
                    db_msas, db_deletion_matrices, db_names = zip(
                        *unsorted_results)
                    msas.append(db_msas)
                    deletion_matrices.append(db_deletion_matrices)
                    names.append(db_names)

            logging.info(f'Making msa pickle: {pickled_msa_path}')
            with open(pickled_msa_path, 'wb') as F:
                pickle.dump(
                    {
                        "msas": msas,
                        "deletion_matrices": deletion_matrices,
                        "names": names
                    }, F)
        return None

    def combine_msas(self,
                     input_fasta_path: str,
                     msa_output_dir: str,
                     homooligomer: str,
                     turbo: bool = False) -> None:
        """Runs alignment tools on the input sequence. Note, input sequence should be ori_sequence"""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]

        ori_sequence: str = input_sequence  # ori_sequence = "MLASVAS:ASVASDV"
        sequence: str = ori_sequence.replace(':',
                                             '')  # sequence = "MLASVASASVASDV"
        seqs: List[str] = ori_sequence.split(
            ':')  # seqs = ["MLASVAS", "ASVASDV"]
        homooligomers: List[int] = [int(h) for h in homooligomer.split(':')
                                    ]  # homooligomers = [2, 2]
        full_sequence: str = "".join([
            s * h for s, h in zip(seqs, homooligomers)
        ])  # full_sequence = "MLASVASMLASVASASVASDVASVASDV"

        _blank_seq = ["-" * len(seq) for seq in seqs]
        _blank_mtx = [[0] * len(seq) for seq in seqs]

        def _pad(ns, vals, mode):
            if mode == "seq": _blank = _blank_seq.copy()
            if mode == "mtx": _blank = _blank_mtx.copy()
            if isinstance(ns, list):
                for n, val in zip(ns, vals):
                    _blank[n] = val
            else:
                _blank[ns] = vals
            if mode == "seq": return "".join(_blank)
            if mode == "mtx": return sum(_blank, [])

        combined_msa_pickle = os.path.join(msa_output_dir,
                                           'combined_msa.pickle')
        if not os.path.exists(combined_msa_pickle):
            msas = []
            deletion_matrices = []
            for n, seq in enumerate(seqs):
                prefix = str(n)
                pickled_msa_path = os.path.join(msa_output_dir,
                                                f"{prefix}.pickle")
                msas_dict = pickle.load(open(pickled_msa_path, "rb"))
                msas_, mtxs_, names_ = (
                    msas_dict[k]
                    for k in ['msas', 'deletion_matrices', 'names'])
                # pad sequences
                for msa_, mtx_ in zip(msas_, mtxs_):
                    msa, mtx = [sequence], [[0] * len(sequence)]
                    for s, m in zip(msa_, mtx_):
                        msa.append(_pad(n, s, "seq"))
                        mtx.append(_pad(n, m, "mtx"))

                    msas.append(msa)
                    deletion_matrices.append(mtx)
            pickle.dump({
                "msas": msas,
                "deletion_matrices": deletion_matrices
            }, open(combined_msa_pickle, "wb"))
        else:
            with open(combined_msa_pickle, 'rb') as F:
                combined_msa = pickle.load(F)
            msas = combined_msa['msas']
            deletion_matrices = combined_msa['deletion_matrices']

        full_msa = []
        for msa in msas:
            full_msa += msa
        deduped_full_msa = list(dict.fromkeys(full_msa))
        total_msa_size = len(deduped_full_msa)
        logging.info(f'{total_msa_size} Sequences Found in Total\n')

        lengths = [len(seq) for seq in seqs]
        msas_mod, deletion_matrices_mod = homooligomerize_heterooligomer(
            msas, deletion_matrices, lengths, homooligomers)

        num_res = len(full_sequence)
        feature_dict = {}
        feature_dict.update(
            make_sequence_features(full_sequence, 'test', num_res))
        feature_dict.update(
            make_msa_features(msas_mod,
                              deletion_matrices=deletion_matrices_mod,
                              Ln=num_res))
        if not turbo:
            feature_dict.update(_placeholder_template_feats(0, num_res))

        logging.info(f'Sequences: {str(seqs)}')
        logging.info(f'Homooligomers: {str(homooligomers)}')
        Ls = []
        for seq, h in zip(seqs, homooligomers):
            Ls += [len(seq)] * h
        logging.info(
            f'Original residue index: {str(feature_dict["residue_index"])}')
        logging.info(f'Sequence Lengths: {str(Ls)}')
        feature_dict['residue_index'] = chain_break(
            feature_dict['residue_index'], Ls)
        logging.info(
            f'Fixed residue index: {str(feature_dict["residue_index"])}')

        logging.info('Final (deduplicated) MSA size: %d sequences.',
                     feature_dict['num_alignments'][0])

        return feature_dict
