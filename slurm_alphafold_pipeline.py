#!/usr/bin/env python
"""This CLI tool is meant to run alphafold using slurm lawrencium
resources. Tools for protein complex modeling were adapted from
https://github.com/sokrypton/ColabFold"""
__VERSION__ = "8/21/21"
__AUTHOR__ = "Alberto Nava <aanava@lbl.gov>"

# =======================================================================#
# Importations
# =======================================================================#

# CLI Template Imports. Do not remove
import argparse
import logging

# Native Python libraries
import os
import shutil
import subprocess
from typing import List, Tuple, Sequence

# =======================================================================#
# Command-Line Interface
# =======================================================================#


def cli() -> argparse.ArgumentParser:
    """ Command-Line Interface Function

    Arguments
    ---------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        A command line parser object
    """
    parser = argparse.ArgumentParser(
        description=("A CLI for running alphafold on lawrencium"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__VERSION__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-m",
        "--model_only",
        action='store_true',
        help=('Whether to just run model stage (GPU stage)'),
    )
    parser.add_argument(
        "-f",
        "--features_only",
        action='store_true',
        help=('Whether to just run features stage (CPU stage)'),
    )
    parser.add_argument(
        "-b",
        "--block",
        action='store_true',
        help=('Whether to block command line after launching slurm job,'
              ' e.g. use --wait option. If -b given then will use --wait. '
              'Default is to submit to queue and release'),
    )
    parser.add_argument(
        "-n",
        "--num_models",
        type=int,
        default=5,
        help=('Number of models to make for each target_fasta'),
    )
    parser.add_argument(
        "-r",
        "--relax",
        action='store_true',
        help=('Whether to relax structures with Amber Molecular Dynamics. '
              'Note, cannot relax when homooligomers > 1'),
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default="reduced_dbs",
        choices=["reduced_dbs", "full_dbs"],
        help=('Alphafold database preset: reduced_dbs or full_dbs'),
    )
    parser.add_argument(
        "-H",
        "--homooligomers",
        type=str,
        default='1',
        help=('From ColabFold: Define number of copies in homo-oligomeric '
              'assembly. Use : to specify different homooligomeric state '
              'for each component. For example, sequence:ABC:DEF, '
              'homooligomer: 2:1, the first protein ABC will be modeled as '
              'a homodimer (2 copies) and second DEF a monomer (1 copy).'),
    )
    parser.add_argument(
        "--mmseqs",
        action='store_true',
        help=('Whether to use mmseqs to create MSAs. MMSeqs is 10x '
              'faster than alphafold method'),
    )
    parser.add_argument(
        "--max_recycles",
        type=int,
        default=3,
        help=('From ColabFold: max_recycles controls the maximum number of '
              'times the structure is fed back into the neural network for '
              'refinement. (3 recommended)'),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help=('From ColabFold: tol tolerance for deciding when to stop '
              'recycles (CA-RMS between recycles)'),
    )
    parser.add_argument(
        "--use_ptm",
        action='store_true',
        help=('From ColabFold: use_ptm uses Deepminds ptm finetuned model '
              'parameters to get PAE per structure. Disable to use the '
              'original model params. (Disabling may give alternative '
              'structures.)'),
    )
    parser.add_argument(
        "--alphafold",
        type=str,
        default="/global/scratch/aanava/alphafold",
        help=(
            'Path to alphafold git repo with non-docker helper shell scripts'),
    )
    parser.add_argument(
        "--alphafold_input",
        type=str,
        default="/global/scratch/aanava/alphafold_input",
        help=('Path to folder containing target_fastas'),
    )
    parser.add_argument(
        "--alphafold_databases",
        type=str,
        default="/global/scratch/aanava/alphafold_databases",
        help=('Path to folder containing alphafold databases'),
    )
    parser.add_argument(
        "--alphafold_results",
        type=str,
        default="/global/scratch/aanava/alphafold_results",
        help=('Path to alphafold output folder'),
    )
    parser.add_argument(
        "--miniconda",
        type=str,
        default="/global/scratch/aanava/miniconda3/bin/activate",
        help=('Path to miniconda activate script'),
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default="0",
        help=('CUDA_VISIBLE_DEVICES'),
    )
    parser.add_argument(
        "--turbo",
        action='store_true',
        help=('Whether to use alphafold turbo models'),
    )
    parser.add_argument(
        "target_fastas",
        type=str,
        nargs='+',
        help=("Names of target fastas. Should have one sequence per file. No "
              "stop codons"),
    )
    return parser


def loggingHelper(verbose=False, filename="slurm_alphafold_pipeline.log"):
    """ Helper to set up python logging

    Arguments
    ---------
    verbose : bool, optional
        Whether to set up verbose logging [default: False]

    Returns
    -------
    None
        Sets up logging
    """
    if verbose:
        loggingLevel = logging.DEBUG  # show everything
    else:
        loggingLevel = logging.ERROR  # show only ERROR and CRITICAL
    logging.basicConfig(
        filename=filename,
        level=loggingLevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    return None


# =======================================================================#
# SLURM Script Templates
# 2 separate scripts for 2 different stages of alphafold
# 1st stage: Feature generation is CPU limited
# - Run on either lr6 or lr3
# 2nd stage: DNN Structure Model is GPU limited
# - Run on es1
# =======================================================================#

ALPHAFOLD_FEATURE_TEMPLATE: str = """#!/bin/bash
#SBATCH --job-name=alphamsa_{NAME}
#SBATCH --partition=lr6
#SBATCH --account=pc_rosetta
#SBATCH --qos=lr_normal
#SBATCH --output={ALPHAFOLD_LOGS}/%j_%x_{NAME}_{PURPOSE}.out
#SBATCH --error={ALPHAFOLD_LOGS}/%j_%x_{NAME}_{PURPOSE}.err
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem {PARTITION_MEM}

###SBATCH --mem {PARTITION_MEM}
###SBATCH --partition={PARTITION}
###SBATCH --mem {PARTITION_MEM}

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)

source {MINICONDA}
conda activate alphafold
cd {ALPHAFOLD}

fasta_path="{TARGET_FASTA}"
preset="{PRESET}"
homooligomer="{HOMOOLIGOMERS}"
data_dir="{ALPHAFOLD_DATABASES}"
output_dir="{ALPHAFOLD_RESULTS}"
model_names="model_1"
max_template_date="2022-12-31"
benchmark=false
max_recycles={MAX_RECYCLES}
tol={TOL}

alphafold_script="{ALPHAFOLD}/{PURPOSE}.py"
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
mgnify_database_path="$data_dir/mgnify/mgy_clusters.fa"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"
pdb70_database_path="$data_dir/pdb70/pdb70"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)
mmseqs_binary_path=$(which mmseqs)
mmseqs_uniref50_database_path="$data_dir/uniref50/uniref50"
mmseqs_mgnify_database_path="$data_dir/mgnify/mgnify"
mmseqs_small_bfd_database_path="$data_dir/small_bfd/small_bfd"

if [[ $preset == "reduced_dbs" ]]; then
    echo 'Creating reduced_dbs features.pkl'
    $(/usr/bin/time -v python $alphafold_script \
        --hhblits_binary_path=$hhblits_binary_path \
        --hhsearch_binary_path=$hhsearch_binary_path \
        --jackhmmer_binary_path=$jackhmmer_binary_path \
        --kalign_binary_path=$kalign_binary_path \
        --mgnify_database_path=$mgnify_database_path \
        --template_mmcif_dir=$template_mmcif_dir \
        --obsolete_pdbs_path=$obsolete_pdbs_path \
        --pdb70_database_path=$pdb70_database_path \
        --uniref90_database_path=$uniref90_database_path \
        --small_bfd_database_path=$small_bfd_database_path \
        --mmseqs_binary_path=$kalign_binary_path \
        --mmseqs_uniref50_database_path=$mmseqs_uniref50_database_path \
        --mmseqs_mgnify_database_path=$mmseqs_mgnify_database_path \
        --mmseqs_small_bfd_database_path=$mmseqs_small_bfd_database_path \
        --data_dir=$data_dir \
        --output_dir=$output_dir \
        --fasta_paths=$fasta_path \
        --model_names=$model_names \
        --max_template_date=$max_template_date \
        --preset=$preset \
        --benchmark=$benchmark \
        --logtostderr \
        --homooligomer=$homooligomer \
        --max_recycles=$max_recycles \
        --tol=$tol \
        {COMPLEX_NAME} \
        {MMSEQS} \
        {TURBO})
else
    echo 'Creating full_dbs features.pkl'
    $(/usr/bin/time -v python $alphafold_script \
        --hhblits_binary_path=$hhblits_binary_path \
        --hhsearch_binary_path=$hhsearch_binary_path \
        --jackhmmer_binary_path=$jackhmmer_binary_path \
        --kalign_binary_path=$kalign_binary_path \
        --bfd_database_path=$bfd_database_path \
        --mmseqs_binary_path=$kalign_binary_path \
        --mmseqs_uniref50_database_path=$mmseqs_uniref50_database_path \
        --mmseqs_mgnify_database_path=$mmseqs_mgnify_database_path \
        --mmseqs_small_bfd_database_path=$mmseqs_small_bfd_database_path \
        --mgnify_database_path=$mgnify_database_path \
        --template_mmcif_dir=$template_mmcif_dir \
        --obsolete_pdbs_path=$obsolete_pdbs_path \
        --pdb70_database_path=$pdb70_database_path \
        --uniclust30_database_path=$uniclust30_database_path \
        --uniref90_database_path=$uniref90_database_path \
        --data_dir=$data_dir \
        --output_dir=$output_dir \
        --fasta_paths=$fasta_path \
        --model_names=$model_names \
        --max_template_date=$max_template_date \
        --preset=$preset \
        --benchmark=$benchmark \
        --logtostderr \
        --homooligomer=$homooligomer \
        --max_recycles=$max_recycles \
        --tol=$tol \
        {COMPLEX_NAME} \
        {MMSEQS} \
        {TURBO})
fi
"""

ALPHAFOLD_MODEL_TEMPLATE: str = """#!/bin/bash
#SBATCH --job-name=alphamodel_{NAME}
#SBATCH --partition=es1
#SBATCH --account=pc_rosetta
#SBATCH --qos=es_normal
#SBATCH --output={ALPHAFOLD_LOGS}/%j_%x_{NAME}.out
#SBATCH --error={ALPHAFOLD_LOGS}/%j_%x_{NAME}.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:GTX1080TI:2
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive

##SBATCH --gres=gpu:2
##SBATCH --mem 180G
# gpu:V100:2
# gpu:GTX1080TI:4
# gpu:GRTX2080TI:4

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)
nvidia-smi

module purge
module load cuda/10.2

source {MINICONDA}
conda activate alphafold
cd {ALPHAFOLD}

fasta_path="{TARGET_FASTA}"
preset="{PRESET}"
homooligomer="{HOMOOLIGOMERS}"
data_dir="{ALPHAFOLD_DATABASES}"
output_dir="{ALPHAFOLD_RESULTS}"
model_names="{MODELS}"
relax="{RELAX_STRUCTURES}"
max_template_date="2022-12-31"
benchmark=false
max_recycles={MAX_RECYCLES}
tol={TOL}

alphafold_script="{ALPHAFOLD}/run_alphafold.py"
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
mgnify_database_path="$data_dir/mgnify/mgy_clusters.fa"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"
pdb70_database_path="$data_dir/pdb70/pdb70"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)
mmseqs_binary_path=$(which mmseqs)
mmseqs_uniref50_database_path="$data_dir/uniref50/uniref50"
mmseqs_mgnify_database_path="$data_dir/mgnify/mgnify"
mmseqs_small_bfd_database_path="$data_dir/small_bfd/small_bfd"

export CUDA_VISIBLE_DEVICES={GPU_DEVICES}
export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

if [[ $preset == "reduced_dbs" ]]; then
    echo 'Running alphafold'
    $(/usr/bin/time -v python $alphafold_script \
        --hhblits_binary_path=$hhblits_binary_path \
        --hhsearch_binary_path=$hhsearch_binary_path \
        --jackhmmer_binary_path=$jackhmmer_binary_path \
        --kalign_binary_path=$kalign_binary_path \
        --mgnify_database_path=$mgnify_database_path \
        --template_mmcif_dir=$template_mmcif_dir \
        --obsolete_pdbs_path=$obsolete_pdbs_path \
        --pdb70_database_path=$pdb70_database_path \
        --uniref90_database_path=$uniref90_database_path \
        --small_bfd_database_path=$small_bfd_database_path \
        --mmseqs_binary_path=$kalign_binary_path \
        --mmseqs_uniref50_database_path=$mmseqs_uniref50_database_path \
        --mmseqs_mgnify_database_path=$mmseqs_mgnify_database_path \
        --mmseqs_small_bfd_database_path=$mmseqs_small_bfd_database_path \
        --data_dir=$data_dir \
        --output_dir=$output_dir \
        --fasta_paths=$fasta_path \
        --model_names=$model_names \
        --max_template_date=$max_template_date \
        --preset=$preset \
        --benchmark=$benchmark \
        --logtostderr \
        --homooligomer=$homooligomer \
        --relax=$relax \
        --max_recycles=$max_recycles \
        --tol=$tol \
        {MMSEQS} \
        {TURBO})
else
    echo 'Running alphafold'
    $(/usr/bin/time -v python $alphafold_script \
        --hhblits_binary_path=$hhblits_binary_path \
        --hhsearch_binary_path=$hhsearch_binary_path \
        --jackhmmer_binary_path=$jackhmmer_binary_path \
        --kalign_binary_path=$kalign_binary_path \
        --bfd_database_path=$bfd_database_path \
        --mmseqs_binary_path=$kalign_binary_path \
        --mmseqs_uniref50_database_path=$mmseqs_uniref50_database_path \
        --mmseqs_mgnify_database_path=$mmseqs_mgnify_database_path \
        --mmseqs_small_bfd_database_path=$mmseqs_small_bfd_database_path \
        --mgnify_database_path=$mgnify_database_path \
        --template_mmcif_dir=$template_mmcif_dir \
        --obsolete_pdbs_path=$obsolete_pdbs_path \
        --pdb70_database_path=$pdb70_database_path \
        --uniclust30_database_path=$uniclust30_database_path \
        --uniref90_database_path=$uniref90_database_path \
        --data_dir=$data_dir \
        --output_dir=$output_dir \
        --fasta_paths=$fasta_path \
        --model_names=$model_names \
        --max_template_date=$max_template_date \
        --preset=$preset \
        --benchmark=$benchmark \
        --logtostderr \
        --homooligomer=$homooligomer \
        --relax=$relax \
        --max_recycles=$max_recycles \
        --tol=$tol \
        {MMSEQS} \
        {TURBO})
fi
"""

# =======================================================================#
# Main
# =======================================================================#


def create_combine_msa_script(target_fasta: str, args: dict,
                              complex_name: str) -> str:
    name: str = os.path.splitext(os.path.basename(target_fasta))[0]
    output_dir: str = os.path.join(args['alphafold_results'], complex_name)
    logs_dir: str = os.path.join(output_dir, 'logs')
    msa_script: str = ALPHAFOLD_FEATURE_TEMPLATE.format(
        **{
            "TARGET_FASTA": target_fasta,
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_LOGS": logs_dir,
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
            "PURPOSE": 'run_combine_msas',
            "MAX_RECYCLES": args['max_recycles'],
            "TOL": args['tol'],
            "COMPLEX_NAME": f'--complex_name={complex_name}',
            "PARTITION": 'lr6',
            "PARTITION_MEM": '48G',
            "TURBO": f'--turbo={args["turbo"]}',
            "MMSEQS": '',
        })
    msa_script_path: str = os.path.join(
        output_dir, f'submit_combine_msa_{complex_name}.slurm')
    with open(msa_script_path, 'w') as F:
        F.write(msa_script)
    return msa_script_path


def create_msa_script(target_fasta: str, args: dict, complex_name: str) -> str:
    name: str = os.path.splitext(os.path.basename(target_fasta))[0]
    output_dir: str = os.path.join(args['alphafold_results'], complex_name)
    logs_dir: str = os.path.join(output_dir, 'logs')
    msa_script: str = ALPHAFOLD_FEATURE_TEMPLATE.format(
        **{
            "TARGET_FASTA": os.path.join(args['alphafold_input'],
                                         target_fasta),
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_LOGS": logs_dir,
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
            "PURPOSE": 'run_msas',
            "MAX_RECYCLES": args['max_recycles'],
            "TOL": args['tol'],
            "COMPLEX_NAME": f'--complex_name={complex_name}',
            "PARTITION": 'lr6',
            "PARTITION_MEM": '180G',
            "TURBO": '',
            "MMSEQS": f'--mmseqs={args["mmseqs"]}',
        })
    msa_script_path: str = os.path.join(
        output_dir, f'submit_create_msa_{complex_name}_{name}.slurm')
    with open(msa_script_path, 'w') as F:
        F.write(msa_script)
    return msa_script_path


def create_feature_script(target_fasta: str, args: dict) -> str:
    name: str = os.path.splitext(os.path.basename(target_fasta))[0]
    output_dir: str = os.path.join(args['alphafold_results'], name)
    logs_dir: str = os.path.join(output_dir, 'logs')
    feature_script: str = ALPHAFOLD_FEATURE_TEMPLATE.format(
        **{
            "TARGET_FASTA": os.path.join(args['alphafold_input'],
                                         target_fasta),
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_LOGS": logs_dir,
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
            "PURPOSE": 'run_feature',
            "MAX_RECYCLES": args['max_recycles'],
            "TOL": args['tol'],
            "COMPLEX_NAME": '',
            "PARTITION": 'lr6',
            "PARTITION_MEM": '180G',
            "TURBO": '',
            "MMSEQS": '',
        })
    feature_script_path: str = os.path.join(output_dir,
                                            f'submit_features_{name}.slurm')
    with open(feature_script_path, 'w') as F:
        F.write(feature_script)
    return feature_script_path


def create_model_script(target_fasta: str, args: dict) -> str:
    name: str = os.path.splitext(os.path.basename(target_fasta))[0]
    if args['use_ptm']:
        models: str = ','.join(
            [f'model_{i}_ptm' for i in range(1, args['num_models'] + 1)])
    else:
        models: str = ','.join(
            [f'model_{i}' for i in range(1, args['num_models'] + 1)])
    relax: str = 'true' if args['relax'] else 'false'
    output_dir: str = os.path.join(args['alphafold_results'], name)
    logs_dir: str = os.path.join(output_dir, 'logs')
    model_script: str = ALPHAFOLD_MODEL_TEMPLATE.format(
        **{
            "TARGET_FASTA": os.path.join(args['alphafold_input'],
                                         target_fasta),
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MODELS": models,
            "RELAX_STRUCTURES": relax,
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_LOGS": logs_dir,
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
            "GPU_DEVICES": args['gpu_devices'],
            "MAX_RECYCLES": args['max_recycles'],
            "TOL": args['tol'],
            "TURBO": f'--turbo={args["turbo"]}',
            "MMSEQS": '',
        })
    model_script_path: str = os.path.join(output_dir,
                                          f'submit_models_{name}.slurm')
    with open(model_script_path, 'w') as F:
        F.write(model_script)
    return model_script_path


def launch_slurm_process(command: str) -> str:
    logging.debug(command)
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        check=True,
    )
    process_id: str = process.stdout.decode().strip().split(' ')[-1]
    return process_id


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Notes:
      Function from deepmind alphafold

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def main(args: dict) -> None:
    """ Command-Line Main Function

    Arguments
    ---------
    args : dict
        CLI-interface options from argparse

    Returns
    -------
    outputFilename : str
        Path to alignment input fasta file that was created
    """
    logging.debug(args)
    assert os.path.exists(args['miniconda'])
    assert os.path.isdir(args['alphafold'])
    assert os.path.isdir(args['alphafold_input'])
    assert os.path.isdir(args['alphafold_databases'])
    assert os.path.isdir(args['alphafold_results'])
    for target_fasta in args['target_fastas']:
        assert os.path.exists(
            os.path.join(args['alphafold_input'], target_fasta))

    logging.debug("Beginning to run alphafold")

    for target_fasta in args['target_fastas']:
        logging.debug(f"Setting up {target_fasta}")

        with open(os.path.join(args['alphafold_input'], target_fasta),
                  'r') as F:
            raw_fasta: str = F.read()
        ori_sequence: str = parse_fasta(raw_fasta)[0][
            0]  # ori_sequence = "MLASVAS:ASVASDV"
        seqs: List[str] = ori_sequence.split(
            ':')  # seqs = ["MLASVAS", "ASVASDV"]
        homooligomer: str = args['homooligomers']  # homooligomer = "2:2"
        homooligomers: List[int] = [
            int(h) for h in args['homooligomers'].split(':')
        ]  # homooligomers = [2, 2]
        assert len(homooligomers) == 1 or len(homooligomers) == len(
            seqs
        ), f'Homooligomers: {str(homooligomers)} vs len(seqs): {len(seqs)}'
        if len(homooligomers) == 1 and len(seqs) > 1:
            homooligomers *= len(seqs)
            homooligomer = ':'.join(str(h) for h in homooligomers)
        full_sequence: str = "".join([
            s * h for s, h in zip(seqs, homooligomers)
        ])  # full_sequence = "MLASVASMLASVASASVASDVASVASDV"
        logging.debug(f"Original homooligomer: {args['homooligomers']}")
        logging.debug(f"Parsed homooligomer: {homooligomer}")
        logging.debug(f"Parsed homooligomers: {str(homooligomers)}")
        logging.debug(f"Original Sequence: {ori_sequence}")
        logging.debug(f"Split Sequences: {str(seqs)}")
        logging.debug(f"Full Sequence: {full_sequence}")
        logging.debug(f"Total length of {target_fasta}: {len(full_sequence)}")
        args['homooligomers'] = homooligomer

        name: str = os.path.splitext(target_fasta)[0]
        output_directory: str = os.path.join(args['alphafold_results'], name)
        if os.path.exists(os.path.join(output_directory, 'ranked_0.pdb')):
            logging.debug(f'Skipping {name} because ranked_0.pdb exists')
            continue
        msas_directory: str = os.path.join(output_directory, 'msas')
        logs_directory: str = os.path.join(output_directory, 'logs')
        os.makedirs(msas_directory, exist_ok=True)
        os.makedirs(logs_directory, exist_ok=True)
        if not os.path.exists(os.path.join(output_directory, target_fasta)):
            shutil.copy2(os.path.join(args['alphafold_input'], target_fasta),
                         os.path.join(output_directory, target_fasta))

        scommand: str = 'sbatch --wait' if args['block'] else 'sbatch'
        logging.debug(f'Using scommand: {scommand}')

        model_script_path: str
        if len(seqs) == 1:
            assert len(homooligomers) == 1
            feature_script_path: str = create_feature_script(
                target_fasta=os.path.join(args['alphafold_input'],
                                          target_fasta),
                args=args)
            model_script_path = create_model_script(target_fasta=os.path.join(
                args['alphafold_input'], target_fasta),
                                                    args=args)

            model_process_id: str
            if not args['model_only'] and not args['features_only']:
                feature_process_id: str = launch_slurm_process(
                    f"{scommand} {feature_script_path}")

                model_process_id = launch_slurm_process(
                    f"{scommand} --dependency=afterok:{feature_process_id} {model_script_path}"  # noqa: E501
                )

                logging.debug(
                    f"Launched feature process {feature_process_id} and "
                    f"dependent model process {model_process_id} for "
                    f"{target_fasta}")
            elif args['features_only']:
                feature_process_id: str = launch_slurm_process(
                    f"{scommand} {feature_script_path}")
                logging.debug(
                    f"Launched feature process {feature_process_id} for "
                    f"{target_fasta}")
            elif args['model_only']:
                model_process_id = launch_slurm_process(
                    f"{scommand} {model_script_path}")

                logging.debug(f"Launched model process {model_process_id} for "
                              f"{target_fasta}")
        else:
            msa_scripts: List[str] = []
            for i, seq in enumerate(seqs):
                tmp_seq_path = os.path.join(msas_directory, f'{i}.fasta')
                with open(tmp_seq_path, 'w') as F:
                    F.write('>{}\n{}'.format(i, seq))
                msa_script_path: str = create_msa_script(
                    target_fasta=tmp_seq_path, args=args, complex_name=name)
                msa_scripts.append(msa_script_path)
            combine_msa_script_path: str = create_combine_msa_script(
                target_fasta=os.path.join(args['alphafold_input'],
                                          target_fasta),
                args=args,
                complex_name=name)
            model_script_path = create_model_script(target_fasta=os.path.join(
                args['alphafold_input'], target_fasta),
                                                    args=args)
            model_script_id: str
            if not args['model_only'] and not args['features_only']:
                msa_script_ids: List[str] = []
                for msa_script in msa_scripts:
                    msa_script_id: str = launch_slurm_process(
                        f"{scommand} {msa_script}")
                    msa_script_ids.append(msa_script_id)
                combine_msa_script_id: str = launch_slurm_process(
                    f"{scommand} --dependency=afterok:{':'.join(msa_script_ids)} {combine_msa_script_path}"  # noqa: E501
                )
                model_script_id = launch_slurm_process(
                    f"{scommand} --dependency=afterok:{combine_msa_script_id} {model_script_path}"  # noqa: E501
                )
                logging.debug(
                    f"Launched msa processes {str(msa_script_ids)} and"
                    f" dependent combine msa process {combine_msa_script_id} "
                    f"and dependent model process {model_script_id} for "
                    f"{target_fasta}")
            elif args['features_only']:
                msa_script_ids: List[str] = []
                for msa_script in msa_scripts:
                    msa_script_id: str = launch_slurm_process(
                        f"{scommand} {msa_script}")
                    msa_script_ids.append(msa_script_id)
                combine_msa_script_id: str = launch_slurm_process(
                    f"{scommand} --dependency=afterok:{':'.join(msa_script_ids)} {combine_msa_script_path}"  # noqa: E501
                )
                logging.debug(
                    f"Launched msa processes {str(msa_script_ids)} and"
                    f" dependent combine msa process {combine_msa_script_id} "
                    f"for {target_fasta}")
            elif args['model_only']:
                model_script_id = launch_slurm_process(
                    f"{scommand} {model_script_path}")
                logging.debug(f"Launched model process {model_script_id} for "
                              f"{target_fasta}")

        logging.debug(f"Finished setting up {target_fasta}")

    logging.debug("Finished running alphafold")
    return None


if __name__ == "__main__":
    args: dict = vars(cli().parse_args())
    loggingHelper(verbose=args["verbose"])
    main(args)
