#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --job-name=sk_origa_batch8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --output=sk_origa_batch8.%j.out
#SBATCH --error=sk_origa_batch8%.err

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
combined_mean="0.768 0.476 0.290"
combined_std="0.220 0.198 0.166"

OUTPUT_DIRECTORY="/projects/uthakuria@xsede.org/glaucoma_seg_code/train_outputs/sk_origa_batch8"

echo "Starting training!"

# train
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/sk_train_mask2former.py \
    --train_csv /projects/uthakuria@xsede.org/glaucoma_seg_code/csvs/origa_train.csv \
    --val_csv /projects/uthakuria@xsede.org/glaucoma_seg_code/csvs/origa_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory $OUTPUT_DIRECTORY \
    --dataset_mean $combined_mean \
    --dataset_std $combined_std \
    --lr 0.00002 \
    --batch_size 8 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 100 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 

echo "-- train job finished! --"

sleep 5

MODEL_DIR="/scratch/alpine/uthakuria@xsede.org/fl_glaucoma_checkpoints/sk_origa_batch8/models"
LAST_CHECKPOINT=$(ls ${MODEL_DIR}/model_epoch*_ckpt.pt | sort -V | tail -n )

echo "Using checkpoint: ${LAST_CHECKPOINT}"
echo "Starting testing!"

TEST_OUTPUT_DIRECTORY="/projects/uthakuria@xsede.org/glaucoma_seg_code/test_outputs/sk_origa_batch8"
# test
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/run_inference_multiprocess.py \
    --model_path ${LAST_CHECKPOINT} \
    --input_csv /projects/uthakuria@xsede.org/glaucoma_seg_code/csvs/updated_combined_test.csv \
    --csv_path_col_name image_path \
    --output_root_dir $TEST_OUTPUT_DIRECTORY \
    --num_processes 1 \
    --cuda_num 0

sleep 5

echo "Starting evaluation for disc segmentation!"
# To eval disc:
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/evaluate_jaccard_dice_institution.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder /projects/uthakuria@xsede.org/glaucoma_seg_datasets/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --eval_disc \
    --cuda_num 0

sleep 5

echo "Starting evaluation for disc segmentation!"
# To eval cup:
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/evaluate_jaccard_dice_institution.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder /projects/uthakuria@xsede.org/glaucoma_seg_datasets/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --cuda_num 0

echo "-- test job finished! --"