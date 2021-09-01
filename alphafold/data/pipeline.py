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
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
import numpy as np
from dataclasses import dataclass
from string import ascii_uppercase
import pickle

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]

@dataclass
class Mock_Template_Result():
    features: FeatureDict

def make_mock_template(query_sequence: str):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
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
        'template_release_date': np.array([f'none'.encode()])}
  return Mock_Template_Result(features=template_features)


def make_sequence_features(
        sequence: str, description: str, num_res: int, homooligomer: int = 1) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  Ln: int = len(sequence)
  Ls: Sequence[int] = [Ln]*homooligomer
  num_res = num_res * homooligomer
  sequence = sequence * homooligomer

  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)

  if homooligomer > 1:
    # add big enough number to residue index to indicate chain breaks between oligomers
    idx_res = features['residue_index']
    L_prev = 0
    # Ls: number of residues in each chain
    for L_i in Ls[:-1]:
      idx_res[L_prev+L_i:] += 200
      L_prev += L_i  
    chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    features['residue_index'] = idx_res
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix],
    Ln: int,
    homooligomer: int = 1) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  all_msas = []
  all_deletion_matrices = []
  if homooligomer > 1:
    for o in range(homooligomer):
      for msa, deletion_matrix in zip(msas, deletion_matrices):
        L = Ln * o
        R = Ln * (homooligomer - (o+1))
        all_msas.append(["-"*L+seq+"-"*R for seq in msa])
        all_deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])
  else:
    all_msas = msas
    all_deletion_matrices = deletion_matrices

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  # 1.99 GB Max size, size of row in msa array = Ln * 4 bytes (int32)
  # TODO: CHANGE THIS!!! The extra / 2 was for testing.
  max_msa_sequences = (1.99*1024*1024*1024 / 2) // (Ln * homooligomer * 4) 
  num_sequences = 0
  for msa_index, msa in enumerate(all_msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(all_deletion_matrices[msa_index][sequence_index])
      num_sequences += 1
      if num_sequences >= max_msa_sequences:
        break
    if num_sequences >= max_msa_sequences:
      break

  num_res = len(all_msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


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
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 25000,
               bfd_max_hits: int = 20000):
    """Constructs a feature dict for a given FASTA file."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.bfd_max_hits = bfd_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str, homooligomer: int = 1) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
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
          jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
      hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
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
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
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
      templates_result = make_mock_template(query_sequence=input_sequence*homooligomer)
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
        deletion_matrices=(uniref90_deletion_matrix,
                           bfd_deletion_matrix,
                           mgnify_deletion_matrix),
        Ln=len(input_sequence),
        homooligomer=homooligomer
        )

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
