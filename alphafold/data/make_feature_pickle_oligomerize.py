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

import sys
sys.path.insert(0, '/data/alberto/alphafold')

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
import pickle
from string import ascii_uppercase

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]

def make_sequence_features(
        sequence: str, description: str, num_res: int, Ls: Sequence[int]) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
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

  if len(Ls) > 1:
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
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = dict()
  #print([len(a) for a in deletion_matrix])
  #print(len(deletion_matrix[0]), len(deletion_matrix[-1]))
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features

def pickle_features(feature_dict: FeatureDict, output_dir: str) -> None:
  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)
  return None

def main(
        input_fasta_path: str,
        msa_output_dir: str,
        preset: str = 'reduced_dbs',
        template_max_hits: int = 20,
        mgnify_max_hits: int = 501,
        uniref_max_hits: int = 10000,
        bfd_max_hits: int = 10000,
        homooligomer: int = 1) -> None:
  print('Reading input sequence')
  with open(input_fasta_path, 'r') as f:
    input_fasta_str = f.read()
  input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
  input_sequence = input_seqs[0]*homooligomer
  input_description = input_descs[0]
  Ln = len(input_sequence)
  num_res = len(input_sequence)*homooligomer

  print('Making Sequence Features')
  sequence_features = make_sequence_features(
      sequence=input_sequence,
      description=input_description,
      num_res=num_res,
      Ls=[Ln]*homooligomer
      )

  print('Reading msas')
  uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
  jackhmmer_uniref90_result = dict()
  with open(uniref90_out_path, 'r') as f:
    jackhmmer_uniref90_result['sto'] = f.read()

  mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
  jackhmmer_mgnify_result = dict()
  with open(mgnify_out_path, 'r') as f:
    jackhmmer_mgnify_result['sto'] = f.read()

  pdb70_out_path = os.path.join(msa_output_dir, 'pdb70_hits.hhr')
  hhsearch_result = ''
  with open(pdb70_out_path, 'r') as f:
    hhsearch_result = f.read()

  print('Parsing msas')
  print('Parsing uniref90')
  uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
      jackhmmer_uniref90_result['sto'])
  print('Parsing mgnify')
  mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
      jackhmmer_mgnify_result['sto'])
  print('Parsing hhsearch')
  hhsearch_hits = parsers.parse_hhr(hhsearch_result)
  mgnify_msa = mgnify_msa[:mgnify_max_hits]
  mgnify_deletion_matrix = mgnify_deletion_matrix[:mgnify_max_hits]
  uniref90_msa = uniref90_msa[:uniref_max_hits]
  uniref90_deletion_matrix = uniref90_deletion_matrix[:uniref_max_hits]

  print('Parsing bfd')
  if preset == 'reduced_dbs':
    bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
    jackhmmer_small_bfd_result = dict()
    with open(bfd_out_path, 'r') as f:
      jackhmmer_small_bfd_result['sto'] = f.read()
    bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
        jackhmmer_small_bfd_result['sto'])
  else:
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    hhblits_bfd_uniclust_result = dict()
    with open(bfd_out_path, 'r') as f:
      hhblits_bfd_uniclust_result['a3m'] = f.write()
    bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
        hhblits_bfd_uniclust_result['a3m'])
  bfd_msa = bfd_msa[:bfd_max_hits]
  bfd_deletion_matrix = bfd_deletion_matrix[:bfd_max_hits]

  print('Grabbing Templates')
  template_featurizer = templates.TemplateHitFeaturizer(
    mmcif_dir='/data/alberto/alphafold_databases/pdb_mmcif/mmcif_files',
    max_template_date='2021-12-31',
    max_hits=template_max_hits,
    kalign_binary_path='/data/alberto/miniconda3/envs/alphafold_nodocker/bin/kalign',
    release_dates_path=None,
    obsolete_pdbs_path='/data/alberto/alphafold_databases/pdb_mmcif/obsolete.dat')
  templates_result = template_featurizer.get_templates(
      query_sequence=input_sequence,
      query_pdb_code=None,
      query_release_date=None,
      hits=hhsearch_hits)

  all_msas = []
  all_deletion_matrices = []
  if homooligomer > 1:
    for o in range(homooligomer):
      for msa, deletion_matrix in zip(
              [uniref90_msa, bfd_msa, mgnify_msa],
              [uniref90_deletion_matrix, bfd_deletion_matrix, mgnify_deletion_matrix]):
        L = Ln * o
        R = Ln * (homooligomer - (o+1))
        all_msas.append(["-"*L+seq+"-"*R for seq in msa])
        all_deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])
  else:
    all_msas = [uniref90_msa, bfd_msa, mgnify_msa]
    all_deletion_matrices = [uniref90_deletion_matrix, bfd_deletion_matrix, mgnify_deletion_matrix]

  print('Making MSA Features')
  msa_features = make_msa_features(
      msas=all_msas,
      deletion_matrices=all_deletion_matrices)

  logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
  logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
  logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
  logging.info('Final (deduplicated) MSA size: %d sequences.',
               msa_features['num_alignments'][0])
  logging.info('Total number of templates (NB: this can include bad '
               'templates and is later filtered to top 4): %d.',
               templates_result.features['template_domain_names'].shape[0])

  print('Writing Pickle')
  final_features = {**sequence_features, **msa_features, **templates_result.features}
  pickle_features(feature_dict=final_features, output_dir=os.path.dirname(msa_output_dir))
  return None

if __name__ == '__main__':
  main(
        input_fasta_path=sys.argv[1],
        msa_output_dir=sys.argv[2],
        preset='reduced_dbs',
        template_max_hits=20,
        mgnify_max_hits=501,
        #uniref_max_hits=200000,
        #bfd_max_hits=25000,
        uniref_max_hits=100000,
        bfd_max_hits=25000,
        homooligomer=2)
