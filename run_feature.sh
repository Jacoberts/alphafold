#!/bin/bash
# Description: AlphaFold non-docker version
# Author: Sanjay Kumar Srikakulam
# From https://github.com/kalininalab/alphafold_non_docker

usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-d <data_dir>     Path to directory of supporting data"
        echo "-o <output_dir>   Path to a directory that will store the results."
        echo "-m <model_names>  Names of models to use (a comma separated list)"
        echo "-f <fasta_path>   Path to a FASTA file containing one sequence"
        echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo "Optional Parameters:"
        echo "-b <benchmark>    Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many
    proteins (default: 'False')"
        echo "-g <use_gpu>      Enable NVIDIA runtime to run with GPUs (default: True)"
        echo "-a <gpu_devices>  Comma separated list of devices to pass to 'CUDA_VISIBLE_DEVICES' (default: 0)"
        echo "-p <preset>       Choose preset model configuration - no ensembling (full_dbs) or 8 model ensemblings (casp14) (default: 'full_dbs')"
        echo "-r <relax>        Whether to relax alphafold structures with amber MD (default: false)"
        echo "-h <homooligomer> How many oligomers to model protein as. Default is to model as monomer (default: 1)"
        echo ""
        exit 1
}


while getopts ":d:o:m:f:t:a:p:g:b:r:h" i; do
        case "${i}" in
        d)
                data_dir=$OPTARG
        ;;
        o)
                output_dir=$OPTARG
        ;;
        m)
                model_names=$OPTARG
        ;;
        f)
                fasta_path=$OPTARG
        ;;
        t)
                max_template_date=$OPTARG
        ;;
        b)
                benchmark=true
        ;;
        g)
                use_gpu=$OPTARG
        ;;
        a)
                gpu_devices=$OPTARG
        ;;
        p)
                preset=$OPTARG
        ;;
        r)
                relax=${OPTARG}
        ;;
        h)
                homooligomer=${OPTARG}
        ;;
        esac
done

# Parse input and set defaults
if [[ "$data_dir" == "" || "$output_dir" == "" || "$model_names" == "" || "$fasta_path" == "" || "$max_template_date" == "" ]] ; then
    usage
fi

if [[ "$benchmark" == "" ]] ; then
    benchmark=false
fi

if [[ "$use_gpu" == "" ]] ; then
    use_gpu=true
fi

if [[ "$gpu_devices" == "" ]] ; then
    gpu_devices=0
fi

if [[ "$preset" == "" ]] ; then
    preset="full_dbs"
fi

if [[ "$relax" == "" ]] ; then
    relax=false
fi

if [[ "$homooligomer" == "" ]] ; then
    homooligomer=1
fi
#homooligomer=2
#gpu_devices=0

# This bash script looks for the run_alphafold.py script in its current working directory, if it does not exist then exits
#current_working_dir=$(pwd)
#alphafold_script="$current_working_dir/run_alphafold.py"
#alphafold_norelax_script="$current_working_dir/run_alphafold_norelax.py"
alphafold_script="run_feature.py"

if [ ! -f "$alphafold_script" ]; then
    echo "Alphafold python script $alphafold_script does not exist."
    exit 1
fi

# Export ENVIRONMENT variables and set CUDA devices for use
export CUDA_VISIBLE_DEVICES=-1
if [[ "$use_gpu" == true ]] ; then
    export CUDA_VISIBLE_DEVICES=0

    if [[ "$gpu_devices" ]] ; then
        export CUDA_VISIBLE_DEVICES=$gpu_devices
    fi
fi

export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

# Path and user config (change me if required)
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
mgnify_database_path="$data_dir/mgnify/mgy_clusters.fa"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"
pdb70_database_path="$data_dir/pdb70/pdb70"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"

# Binary path (change me if required)
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)

# Run AlphaFold with required parameters
if [[ $preset == "reduced_dbs" ]]; then
	echo 'Starting alphafold with reduced_dbs'
	$(python $alphafold_script --hhblits_binary_path=$hhblits_binary_path --hhsearch_binary_path=$hhsearch_binary_path --jackhmmer_binary_path=$jackhmmer_binary_path --kalign_binary_path=$kalign_binary_path --mgnify_database_path=$mgnify_database_path --template_mmcif_dir=$template_mmcif_dir --obsolete_pdbs_path=$obsolete_pdbs_path --pdb70_database_path=$pdb70_database_path --uniref90_database_path=$uniref90_database_path --small_bfd_database_path=$small_bfd_database_path --data_dir=$data_dir --output_dir=$output_dir --fasta_paths=$fasta_path --model_names=$model_names --max_template_date=$max_template_date --preset=$preset --benchmark=$benchmark --logtostderr --homooligomer=$homooligomer)
else
	echo 'Starting alphafold with full_dbs'
	$(python $alphafold_script --hhblits_binary_path=$hhblits_binary_path --hhsearch_binary_path=$hhsearch_binary_path --jackhmmer_binary_path=$jackhmmer_binary_path --kalign_binary_path=$kalign_binary_path --bfd_database_path=$bfd_database_path --mgnify_database_path=$mgnify_database_path --template_mmcif_dir=$template_mmcif_dir --obsolete_pdbs_path=$obsolete_pdbs_path --pdb70_database_path=$pdb70_database_path --uniclust30_database_path=$uniclust30_database_path --uniref90_database_path=$uniref90_database_path --data_dir=$data_dir --output_dir=$output_dir --fasta_paths=$fasta_path --model_names=$model_names --max_template_date=$max_template_date --preset=$preset --benchmark=$benchmark --logtostderr --homooligomer=$homooligomer)
fi
