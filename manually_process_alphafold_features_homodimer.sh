#!/bin/bash

fasta_path=$1
msa_dir=$2

time python /data/alberto/alphafold/alphafold/data/make_feature_pickle_oligomerize.py $fasta_path $msa_dir
