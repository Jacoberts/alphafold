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
"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import numpy as np
from pathlib import Path

#import tensorflow.compat.v1 as tf
#
#logging.info("Num GPUs Available: ",
#             len(tf.config.list_physical_devices('GPU')))
#logging.info("Num CPUs Available: ",
#             len(tf.config.list_physical_devices('CPU')))
#tf.config.set_visible_devices([], 'GPU')
#
#config2 = tf.ConfigProto()
#config2.gpu_options.allow_growth = True
#session = tf.Session(config=config2)

import jax
# Internal import (7716).

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing '
    'one sequence. Paths should be separated by commas. '
    'All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for '
    'each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', '/usr/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', '/usr/bin/hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', '/usr/bin/hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/usr/bin/kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string(
    'small_bfd_database_path', None, 'Path to the small '
    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string(
    'template_mmcif_dir', None, 'Path to a directory with '
    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string(
    'max_template_date', None, 'Maximum template release date '
    'to consider. Important if folding historical test sets.')
flags.DEFINE_string(
    'obsolete_pdbs_path', None, 'Path to file containing a '
    'mapping from obsolete PDB IDs to the PDB IDs of their '
    'replacements.')
flags.DEFINE_enum(
    'preset', 'full_dbs', ['reduced_dbs', 'full_dbs', 'casp14'],
    'Choose preset model configuration - no ensembling and '
    'smaller genetic database config (reduced_dbs), no '
    'ensembling and full genetic database config  (full_dbs) or '
    'full genetic database config and 8 model ensemblings '
    '(casp14).')
flags.DEFINE_boolean(
    'benchmark', False, 'Run multiple JAX model evaluations '
    'to obtain a timing that excludes the compilation time, '
    'which should be more indicative of the time required for '
    'inferencing many proteins.')
flags.DEFINE_boolean('relax', False, 'Whether to run amber relaxation')
flags.DEFINE_integer(
    'random_seed', None, 'The random seed for the data '
    'pipeline. By default, this is randomly generated. Note '
    'that even if this is set, Alphafold may still not be '
    'deterministic, because processes like GPU inference are '
    'nondeterministic.')
flags.DEFINE_float(
    'msa_size_gb', 1.99, 'Size of the MSA.')
flags.DEFINE_string(
    'homooligomer', '1', 'The number of oligomers to model '
    'protein with. By default, will model as monomer '
    '(default: 1)')
flags.DEFINE_integer('max_recycles', '3', 'Max recycles')
flags.DEFINE_float('tol', 0.0, 'Max recycle tolerance')
flags.DEFINE_boolean('turbo', False, 'Whether to use turbo alphafold models')
flags.DEFINE_string('mmseqs_binary_path', '/usr/bin/mmseqs',
                    'Path to the mmseqs executable.')
flags.DEFINE_string('mmseqs_uniref50_database_path', None,
                    'Path to the Uniref50 '
                    'database for use by mmseqs.')
flags.DEFINE_string('mmseqs_mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by mmseqs.')
flags.DEFINE_string('mmseqs_small_bfd_database_path', None, 'Path to the BFD '
                    'database for use by mmseqs.')
flags.DEFINE_boolean('mmseqs', False, 'Whether to use mmseqs MSA pipeline')
flags.DEFINE_string('tmp_dir', None, 'Path to the temp directory.')
flags.DEFINE_boolean('clear_gpu', True, 'Whether to clear GPU memory every time.')
FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def _check_flag(flag_name: str, preset: str, should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = 'be' if should_be_set else 'not be'
        raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


def rm(x):
    '''remove data from device'''
    jax.tree_util.tree_map(lambda y: y.device_buffer.delete(), x)


def to(x, device="cpu"):
    '''move data to device'''
    d = jax.devices(device)[0]
    return jax.tree_util.tree_map(lambda y: jax.device_put(y, d), x)


def clear_mem(device="gpu"):
    '''remove all data from device'''
    backend = jax.lib.xla_bridge.get_backend(device)
    for buf in backend.live_buffers():
        buf.delete()


def predict_structure(fasta_path: str,
                      fasta_name: str,
                      output_dir_base: str,
                      data_pipeline: pipeline.DataPipeline,
                      model_runners: Dict[str, model.RunModel],
                      amber_relaxer: relax.AmberRelaxation,
                      benchmark: bool,
                      random_seed: int,
                      msa_size_gb: float,
                      homooligomer: str = '1',
                      relax: bool = False,
                      turbo: bool = False):
    """Predicts structure using AlphaFold for the given sequence."""
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    features_output_path = os.path.join(output_dir, 'features.pkl')
    if not os.path.exists(features_output_path):
        # Get features.
        t_0 = time.time()
        feature_dict = data_pipeline.process(input_fasta_path=fasta_path,
                                             msa_output_dir=msa_output_dir,
                                             msa_size_gb=msa_size_gb,
                                             homooligomer=homooligomer)
        timings['features'] = time.time() - t_0

        # Write out features as a pickled dictionary.
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)
    else:
        with open(features_output_path, 'rb') as f:
            feature_dict = pickle.load(f)
    #logging.info(str(feature_dict))

    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    plddts = {}
    ptms = {}

    if turbo:
        use_ptm = 'ptm' in list(model_runners.keys())[0]
        is_training = False
        max_recycles = FLAGS.max_recycles
        tol = FLAGS.tol
        num_ensemble = 1
        max_msa = "512:1024"
        max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]
        name = "model_5_ptm" if use_ptm else "model_5"
        N = len(feature_dict["msa"])
        L = len(feature_dict["residue_index"])
        compiled = (N, L, use_ptm, max_recycles, tol, num_ensemble, max_msa,
                    is_training)
        logging.info(f'Turbo model: {str(compiled)}')
        if FLAGS.clear_gpu:
            clear_mem("gpu")
        cfg = config.model_config(name)
        cfg.data.common.max_extra_msa = min(N, max_extra_msa)
        cfg.data.eval.max_msa_clusters = min(N, max_msa_clusters)
        cfg.data.common.num_recycle = max_recycles
        cfg.model.num_recycle = max_recycles
        cfg.model.recycle_tol = tol
        cfg.data.eval.num_ensemble = num_ensemble
        params = data.get_model_haiku_params(model_name=name,
                                             data_dir=FLAGS.data_dir)
        turbo_model_runner = model.RunModel(cfg,
                                            params,
                                            is_training=is_training)

    # Run the models.
    for model_name, model_runner in model_runners.items():
        if turbo:
            params = data.get_model_haiku_params(model_name=model_name,
                                                 data_dir=FLAGS.data_dir)
            for k in turbo_model_runner.params.keys():
                turbo_model_runner.params[k] = params[k]
            model_runner = turbo_model_runner
        logging.info('Running model %s', model_name)
        t_0 = time.time()
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0

        t_0 = time.time()
        #prediction_result, (r, t) = model_runner.predict(processed_feature_dict)
        prediction_result, (r, t) = to(
            model_runner.predict(processed_feature_dict), 'cpu')
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?',
            model_name, t_diff)

        if benchmark:
            t_0 = time.time()
            model_runner.predict(processed_feature_dict)
            timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

        # Get mean pLDDT confidence metric.
        plddt = prediction_result['plddt']
        plddts[model_name] = np.mean(plddt)
        if 'ptm' in prediction_result:
            ptms[model_name] = prediction_result['ptm']

        # Save the model outputs.
        result_output_path = os.path.join(output_dir,
                                          f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
            pickle.dump(prediction_result, f, protocol=4)

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(plddt[:, None],
                                    residue_constants.atom_type_num,
                                    axis=-1)
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors)

        unrelaxed_pdb_path = os.path.join(output_dir,
                                          f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))

        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)

        if relax and homooligomer == '1':
            # Relax the prediction.
            t_0 = time.time()
            relaxed_pdb_str, _, _ = amber_relaxer.process(
                prot=unrelaxed_protein)
            timings[f'relax_{model_name}'] = time.time() - t_0

            relaxed_pdbs[model_name] = relaxed_pdb_str

            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(output_dir,
                                               f'relaxed_{model_name}.pdb')
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)
        if turbo:
            del params

    # Rank by pLDDT and write out relaxed PDBs in rank order.
    ranked_order = []
    for idx, (model_name, _) in enumerate(
            sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        try:
            with open(ranked_output_path, 'w') as f:
                f.write(relaxed_pdbs[model_name])
        except:
            with open(ranked_output_path, 'w') as f:
                f.write(unrelaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        use_ptm = 'ptm' in list(model_runners.keys())[0]
        if not use_ptm:
            try:
                f.write(
                    json.dumps({
                        'plddts': plddts,
                        'order': ranked_order
                    },
                               indent=4))
            except:
                f.write(
                    json.dumps(
                        {
                            'plddts': {
                                k: np.array(v).tolist()
                                for k, v in plddts.items()
                            },
                            'order': ranked_order,
                        },
                        indent=4))
        else:
            f.write(
                json.dumps(
                    {
                        'plddts':
                        {k: np.array(v).tolist()
                         for k, v in plddts.items()},
                        'order': ranked_order,
                        'ptms':
                        {k: np.array(v).tolist()
                         for k, v in ptms.items()},
                    },
                    indent=4))

    logging.info('Final timings for %s: %s', fasta_name, timings)

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if FLAGS.clear_gpu: 
        clear_mem('gpu')
    clear_mem('cpu')

    use_small_bfd = FLAGS.preset == 'reduced_dbs'
    _check_flag('small_bfd_database_path',
                FLAGS.preset,
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path',
                FLAGS.preset,
                should_be_set=not use_small_bfd)
    _check_flag('uniclust30_database_path',
                FLAGS.preset,
                should_be_set=not use_small_bfd)

    if FLAGS.preset in ('reduced_dbs', 'full_dbs'):
        num_ensemble = 1
    elif FLAGS.preset == 'casp14':
        num_ensemble = 8

    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')

    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        tmp_dir=Path(FLAGS.tmp_dir))

    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        hhsearch_binary_path=FLAGS.hhsearch_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniclust30_database_path=FLAGS.uniclust30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        pdb70_database_path=FLAGS.pdb70_database_path,
        template_featurizer=template_featurizer,
        mmseqs_binary_path=FLAGS.mmseqs_binary_path,
        mmseqs_uniref50_database_path=FLAGS.mmseqs_uniref50_database_path,
        mmseqs_mgnify_database_path=FLAGS.mmseqs_mgnify_database_path,
        mmseqs_small_bfd_database_path=FLAGS.mmseqs_small_bfd_database_path,
        mmseqs=FLAGS.mmseqs,
        use_small_bfd=use_small_bfd,
        tmp_dir=Path(FLAGS.tmp_dir))

    model_runners = {}
    if not FLAGS.turbo:
        for model_name in FLAGS.model_names:
            model_config = config.model_config(model_name)
            model_config.data.eval.num_ensemble = num_ensemble
            model_config.data.common.num_recycle = FLAGS.max_recycles
            model_config.model.num_recycle = FLAGS.max_recycles
            model_config.model.recycle_tol = FLAGS.tol
            model_params = data.get_model_haiku_params(model_name=model_name,
                                                       data_dir=FLAGS.data_dir)
            model_runner = model.RunModel(model_config, model_params)
            model_runners[model_name] = model_runner
    else:
        for model_name in FLAGS.model_names:
            model_runners[model_name] = None

    logging.info('Have %d models: %s', len(model_runners),
                 list(model_runners.keys()))

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    logging.info('Using random seed %d for the data pipeline', random_seed)

    msa_size_gb = FLAGS.msa_size_gb

    homooligomer = FLAGS.homooligomer
    if homooligomer is None:
        homooligomer = '1'

    # Predict structure for each of the sequences.
    for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
        predict_structure(fasta_path=fasta_path,
                          fasta_name=fasta_name,
                          output_dir_base=FLAGS.output_dir,
                          data_pipeline=data_pipeline,
                          model_runners=model_runners,
                          amber_relaxer=amber_relaxer,
                          benchmark=FLAGS.benchmark,
                          random_seed=random_seed,
                          msa_size_gb=msa_size_gb,
                          homooligomer=homooligomer,
                          relax=FLAGS.relax,
                          turbo=FLAGS.turbo)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        'model_names',
        'data_dir',
        'preset',
        'uniref90_database_path',
        'mgnify_database_path',
        'pdb70_database_path',
        'template_mmcif_dir',
        'max_template_date',
        'obsolete_pdbs_path',
        'relax',
        'tmp_dir'
    ])

    app.run(main)
