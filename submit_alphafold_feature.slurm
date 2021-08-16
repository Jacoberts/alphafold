#!/bin/bash
#SBATCH --job-name=alpha_msa
#SBATCH --partition=lr6
#SBATCH --account=pc_rosetta
#SBATCH --qos=lr_normal
#SBATCH --output=/global/scratch/aanava/alphafold_input/%j_%x.out
#SBATCH --error=/global/scratch/aanava/alphafold_input/%j_%x.err
#SBATCH --time=4:00:00
#SBATCH -N 1

TARGET_FASTA="vinN.fasta"
PRESET="reduced_dbs"
HOMOOLIGOMERS=1

source /global/scratch/aanava/miniconda3/bin/activate
conda activate alphafold
cd /global/scratch/aanava/alphafold

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)

time /bin/bash /global/scratch/aanava/alphafold/run_feature.sh \
    -d /global/scratch/aanava/alphafold_databases \
    -o /global/scratch/aanava/alphafold_results \
    -t 2022-12-31 \
    -m model_1 \
    -f /global/scratch/aanava/alphafold_input/${TARGET_FASTA} \
    -p ${PRESET} \
    -h ${HOMOOLIGOMERS}
