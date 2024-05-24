#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --job-name=train_fl_glaucoma_seg_all_datasets_v1
#SBATCH --gres=gpu
#SBATCH --ntasks=16
#SBATCH --output=train_fl_glaucoma_seg_all_datasets_v1.%j.out
#SBATCH --error=train_fl_glaucoma_seg_all_datasets_v1%.err

module purge
module load anaconda
conda activate python3_10

# variables
binDriOriga_mean="0.88702821 0.5666646 0.29283205"
binDriOriga_std="0.15141807 0.17632471 0.1514714"
riga_mean="0.83512329 0.49202078 0.22558286"
riga_std="0.14766443 0.16120118 0.12717136"
messidor_mean="0.84204773 0.46466044 0.19547342"
messidor_std="0.14704964 0.15650324 0.11095415"
magrabi_mean="0.8586507  0.57209649 0.25212852"
magrabi_std="0.13919874 0.15322957 0.16213763"
binrushed_mean="0.80747745 0.51806518 0.28384791"
binrushed_std="0.14918518 0.15869727 0.12060928"
combined_mean="0.768 0.476 0.289"
combined_std="0.221 0.198 0.165"

# train
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/fl_glaucoma_train.py \
    --train_csv /projects/uthakuria@xsede.org/glaucoma_seg_code/output_csvs/binrushed_train.csv \
    --val_csv /projects/uthakuria@xsede.org/glaucoma_seg_code/output_csvs/binrushed_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /projects/uthakuria@xsede.org/glaucoma_seg_code/train_outputs/train_outputs_fl_finetuned_binrushed_v2 \
    --dataset_mean $binrushed_mean \
    --dataset_std $binrushed_std \
    --lr 0.00003 \
    --batch_size 16 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 100 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 

echo "-- test job finished! --"