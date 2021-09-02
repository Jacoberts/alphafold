#time python docker/run_docker.py \
#	--fasta_paths=SgcE.fasta \
#	--preset=reduced_dbs \
#	--max_template_date=2020-05-14 \
# 	2>&1 |tee ~/out/SgcE.fasta.stdout
# time bash run_feature.sh \
# 	-d /mnt/disks/afdbs \
# 	-o /home/jacobroberts/out \
# 	-t 2022-12-31 \
# 	-m model_1 \
# 	-f SgcE.fasta \
# 	-p reduced_dbs \
# 	-h 1 
# 
time bash run_alphafold.sh \
	-f SgcE.fasta \
	-t 2022-12-31 \
	-d /mnt/disks/afdbs \
	-o /home/jacobroberts/out \
	-m model_1,model_2,model_3,model_4,model_5 \
	-p reduced_dbs \
	-g False \
	-r true \
	-h 1 \
	2>&1 |tee ~/out/SgcE.fasta.stdout
