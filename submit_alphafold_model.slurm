#!/bin/bash
#SBATCH --job-name=alpha_model
#SBATCH --partition=es1
#SBATCH --account=pc_rosetta
#SBATCH --qos=es_normal
#SBATCH --output=/global/scratch/aanava/alphafold_input/%j_%x.out
#SBATCH --error=/global/scratch/aanava/alphafold_input/%j_%x.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

TARGET_FASTA="vinN.fasta"
MODELS="model_1,model_2,model_3,model_4,model_5"
PRESET="reduced_dbs"
RELAX_STRUCTURES=true
HOMOOLIGOMERS=1

module purge
module load cuda/10.2

source /global/scratch/aanava/miniconda3/bin/activate
conda activate alphafold
cd /global/scratch/aanava/alphafold

echo "HOST: " $(hostname)
echo "NCPU: " $(nproc)
echo "RAM:  " $(free -gth | tail -n -1)

time /bin/bash /global/scratch/aanava/alphafold/run_alphafold.sh \
	-d /global/scratch/aanava/alphafold_databases \
	-o /global/scratch/aanava/alphafold_results \
	-t 2022-12-31 \
	-f /global/scratch/aanava/alphafold_input/${TARGET_FASTA} \
	-m ${MODELS} \
	-p ${PRESET} \
	-r ${RELAX_STRUCTURES} \
	-h ${HOMOOLIGOMERS}
