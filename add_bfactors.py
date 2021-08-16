#!/usr/bin/env python

import os
import sys
import numpy as np
import jax
import pickle
import shutil
from typing import Mapping, Any, Optional

sys.path.insert(0, '/data/alberto/alphafold')

from alphafold.common import protein

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

def from_prediction(
        features: FeatureDict,
        result: ModelOutput,
        b_factors: Optional[np.ndarray] = None) -> protein.Protein:
  """Assembles a protein from a prediction.

    Args:
        features: Dictionary holding model inputs.
        result: Dictionary holding model outputs.
        b_factors: (Optional) B-factors to use for the protein.

    Returns:
        A protein instance.
  """
  fold_output = result['structure_module']
  if b_factors is None:
    b_factors = np.zeros_like(fold_output['final_atom_mask'])

  return protein.Protein(
          aatype=features['aatype'][0],
          atom_positions=fold_output['final_atom_positions'],
          atom_mask=fold_output['final_atom_mask'],
          #residue_index=features['residue_index'][0] + 1,
          residue_index=features['residue_index'],
          b_factors=b_factors)

def parse_results(prediction_result, processed_feature_dict):
  b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']
  out = {
        "unrelaxed_protein": from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
        "plddt": prediction_result['plddt'],
        "sco": prediction_result['plddt'].mean(),
        "b_factors": b_factors,
        #"dists": prediction_result["distogram"]["bin_edges"][prediction_result["distogram"]["logits"].argmax(-1)],
        #"adj": jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,prediction_result["distogram"]["bin_edges"] < 8].sum(-1)}
        }
  if "ptm" in prediction_result:
    out.update({
        "pae": prediction_result['predicted_aligned_error'],
        "ptm": prediction_result['ptm']})
  return out

def main(model_path: str) -> None:
  model_dir: str = os.path.dirname(model_path)
  protein_name: str = os.path.basename(model_dir)
  output_protein_path: str = os.path.join(model_dir, f'{protein_name}_confidence.pdb')
  processed_features_path: str = os.path.join(model_dir, 'features.pkl')
  with open(processed_features_path, 'rb') as F:
    processed_feature_dict = pickle.load(F)
  with open(model_path, 'rb') as F:
    prediction_result = pickle.load(F)
  model_results = parse_results(
        prediction_result=prediction_result,
        processed_feature_dict=processed_feature_dict)
  with open(output_protein_path, 'w') as F:
    F.write(protein.to_pdb(model_results['unrelaxed_protein']))
  return None

if __name__ == '__main__':
    main(model_path=os.path.abspath(sys.argv[1]))
