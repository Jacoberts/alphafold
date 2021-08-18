#!/usr/bin/env python

"""This CLI tool is meant to run alphafold using slurm lawrencium
resources"""
__VERSION__ = "8/17/21"
__AUTHOR__ = "Alberto Nava <aanava@lbl.gov>"


# =======================================================================#
# Importations
# =======================================================================#

# CLI Template Imports. Do not remove
import argparse
import logging

# Native Python libraries
import os
import subprocess

# Non-native python libraries

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
        "-n",
        "--num_models",
	type=int,
	default=5,
        help=(
            'Number of models to make for each target_fasta'
       ),
    )
    parser.add_argument(
        "-r",
        "--relax",
	action='store_true',
        help=(
            'Whether to relax structures with Amber Molecular Dynamics. '
            'Note, cannot relax when homooligomers > 1'
       ),
    )
    parser.add_argument(
        "-p",
        "--preset",
	type=str,
	default="reduced_dbs",
	choices=["reduced_dbs", "full_dbs"],
        help=(
            'Alphafold database preset: reduced_dbs or full_dbs'
       ),
    )
    parser.add_argument(
        "-H",
        "--homooligomers",
	type=int,
	default=1,
        help=(
            'Homoligomer state to model protein as, e.g. 1=monomer, 2=dimer'
       ),
    )
    parser.add_argument(
        "--alphafold",
	type=str,
	default="/global/scratch/aanava/alphafold",
        help=(
            'Path to alphafold git repo with non-docker helper shell scripts'
       ),
    )
    parser.add_argument(
        "--alphafold_input",
	type=str,
	default="/global/scratch/aanava/alphafold_input",
        help=(
            'Path to folder containing target_fastas'
       ),
    )
    parser.add_argument(
        "--alphafold_databases",
	type=str,
	default="/global/scratch/aanava/alphafold_databases",
        help=(
            'Path to folder containing alphafold databases'
       ),
    )
    parser.add_argument(
        "--alphafold_results",
	type=str,
	default="/global/scratch/aanava/alphafold_results",
        help=(
            'Path to alphafold output folder'
       ),
    )
    parser.add_argument(
        "--miniconda",
	type=str,
	default="/global/scratch/aanava/miniconda3/bin/activate",
        help=(
            'Path to miniconda activate script'
       ),
    )
    parser.add_argument(
        "target_fastas",
        type=str,
	nargs='+',
        help=(
            "Names of target fastas. Should have one sequence per file. No "
            "stop codons"
        ),
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

# Replace:
# TARGET_FASTA = vinN.fasta
# NAME = vinN
# PRESET = reduced_dbs
# HOMOOLIGOMERS = 1
# MINICONDA = /global/scratch/aanava/miniconda3/bin/activate
# ALPHAFOLD = /global/scratch/aanava/alphafold
# ALPHAFOLD_INPUT = /global/scratch/aanava/alphafold_input
# ALPHAFOLD_DATABASES = /global/scratch/aanava/alphafold_databases
# ALPHAFOLD_RESULTS = /global/scratch/aanava/alphafold_results
ALPHAFOLD_FEATURE_TEMPLATE: str = """#!/bin/bash
#SBATCH --job-name=alphamsa_{NAME}
#SBATCH --partition=lr6
#SBATCH --account=pc_rosetta
#SBATCH --qos=lr_normal
#SBATCH --output={ALPHAFOLD_INPUT}/%j_%x_{NAME}.out
#SBATCH --error={ALPHAFOLD_INPUT}/%j_%x_{NAME}.err
#SBATCH --time=48:00:00
#SBATCH -N 1

TARGET_FASTA="{TARGET_FASTA}"
PRESET="{PRESET}"
HOMOOLIGOMERS={HOMOOLIGOMERS}

source {MINICONDA}
conda activate alphafold
cd {ALPHAFOLD}

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)

time /bin/bash {ALPHAFOLD}/run_feature.sh \\
    -d {ALPHAFOLD_DATABASES} \\
    -o {ALPHAFOLD_RESULTS} \\
    -t 2022-12-31 \\
    -m model_1 \\
    -f {ALPHAFOLD_INPUT}/${{TARGET_FASTA}} \\
    -p ${{PRESET}} \\
    -h ${{HOMOOLIGOMERS}}
"""

# Replace:
# TARGET_FASTA = vinN.fasta
# NAME = vinN
# PRESET = reduced_dbs
# MODELS = model_1,model_2,model_3,model_4,model_5
# RELAX_STRUCTURES = true
# HOMOOLIGOMERS = 1
# MINICONDA = /global/scratch/aanava/miniconda3/bin/activate
# ALPHAFOLD = /global/scratch/aanava/alphafold
# ALPHAFOLD_INPUT = /global/scratch/aanava/alphafold_input
# ALPHAFOLD_DATABASES = /global/scratch/aanava/alphafold_databases
# ALPHAFOLD_RESULTS = /global/scratch/aanava/alphafold_results
ALPHAFOLD_MODEL_TEMPLATE: str = """#!/bin/bash
#SBATCH --job-name=alphamodel_{NAME}
#SBATCH --partition=es1
#SBATCH --account=pc_rosetta
#SBATCH --qos=es_normal
#SBATCH --output={ALPHAFOLD_INPUT}/%j_%x_{NAME}.out
#SBATCH --error={ALPHAFOLD_INPUT}/%j_%x_{NAME}.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

TARGET_FASTA="{TARGET_FASTA}"
MODELS="{MODELS}"
PRESET="{PRESET}"
RELAX_STRUCTURES={RELAX_STRUCTURES}
HOMOOLIGOMERS={HOMOOLIGOMERS}

module purge
module load cuda/10.2

source {MINICONDA}
conda activate alphafold
cd {ALPHAFOLD}

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)

time /bin/bash {ALPHAFOLD}/run_alphafold.sh \\
	-d {ALPHAFOLD_DATABASES} \\
	-o {ALPHAFOLD_RESULTS} \\
	-t 2022-12-31 \\
	-f {ALPHAFOLD_INPUT}/${{TARGET_FASTA}} \\
	-m ${{MODELS}} \\
	-p ${{PRESET}} \\
	-r ${{RELAX_STRUCTURES}} \\
	-h ${{HOMOOLIGOMERS}}
"""

# =======================================================================#
# Main
# =======================================================================#

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
        assert os.path.exists(os.path.join(args['alphafold_input'], target_fasta))

    logging.debug(f"Beginning to run alphafold")

    for target_fasta in args['target_fastas']:	
        logging.debug(f"Setting up {target_fasta}")
        name: str = os.path.splitext(target_fasta)[0]
        models: str = ','.join([f'model_{i}' for i in range(1, args['num_models']+1)])
        relax: str = 'true' if args['relax'] else 'false'
        feature_script: str = ALPHAFOLD_FEATURE_TEMPLATE.format(**{
            "TARGET_FASTA": target_fasta,
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
        })
        model_script: str = ALPHAFOLD_MODEL_TEMPLATE.format(**{
            "TARGET_FASTA": target_fasta,
            "NAME": name,
            "PRESET": args['preset'],
            "HOMOOLIGOMERS": args['homooligomers'],
            "MODELS": models,
            "RELAX_STRUCTURES": relax,
            "MINICONDA": args['miniconda'],
            "ALPHAFOLD": args['alphafold'],
            "ALPHAFOLD_INPUT": args['alphafold_input'],
            "ALPHAFOLD_DATABASES": args['alphafold_databases'],
            "ALPHAFOLD_RESULTS": args['alphafold_results'],
        })

        feature_script_path: str = os.path.join(args['alphafold_input'], f'submit_features_{name}.slurm')
        with open(feature_script_path, 'w') as F:
            F.write(feature_script)

        model_script_path: str = os.path.join(args['alphafold_input'], f'submit_models_{name}.slurm')
        with open(model_script_path, 'w') as F:
            F.write(model_script)
        
        feature_process_command: str = f"sbatch {feature_script_path}"
        logging.debug(feature_process_command)
        feature_process = subprocess.run(
            feature_process_command,
            stdout=subprocess.PIPE,
            shell=True,
            check=True,
        )
        feature_process_id: str = feature_process.stdout.decode().strip().split(' ')[-1]

        model_process_command: str = f"sbatch --dependency=afterok:{feature_process_id} {model_script_path}"
        logging.debug(model_process_command)
        model_process = subprocess.run(
            model_process_command,
            stdout=subprocess.PIPE,
            shell=True,
            check=True,
        )
        model_process_id: str = model_process.stdout.decode().strip().split(' ')[-1]

        logging.debug(f"Launched feature process {feature_process_id} and dependent model process {model_process_id} for {target_fasta}")
        logging.debug(f"Finished setting up {target_fasta}")

    logging.debug(f"Finished running alphafold")
    return None


if __name__ == "__main__":
    args: dict = vars(cli().parse_args())
    loggingHelper(verbose=args["verbose"])
    main(args)
