#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --job-name=sk_train_central_batch8_v1
#SBATCH --gres=gpu
#SBATCH --ntasks=8
#SBATCH --output=sk_train_central_batch8_v1.%j.out
#SBATCH --error=sk_train_central_batch8_v1%.err

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

OUTPUT_DIRECTORY="/projects/uthakuria@xsede.org/glaucoma_seg_code/train_outputs/sk_train_central_batch8_v1"
MODEL_DIR="${OUTPUT_DIRECTORY}/model"
LAST_CHECKPOINT="/projects/uthakuria@xsede.org/glaucoma_seg_code/train_outputs/sk_train_central_batch8_v1/model/model_epoch20_ckpt.pt"

echo "Using checkpoint: ${LAST_CHECKPOINT}"
echo "Starting testing!"

TEST_OUTPUT_DIRECTORY="/projects/uthakuria@xsede.org/glaucoma_seg_code/test_outputs/sk_train_central_batch8_v1"
echo "Starting evaluation for disc segmentation!"
# To eval disc:
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/evaluate_jaccard_dice_institution.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder /projects/uthakuria@xsede.org/glaucoma_seg_datasets/ \
    --csv_path /projects/uthakuria@xsede.org/glaucoma_seg_code/test_outputs/sk_train_central_batch8_v1/results.csv \
    --eval_disc \
    --cuda_num 0

echo "Starting evaluation for disc segmentation!"
# To eval cup:
python3 /projects/uthakuria@xsede.org/glaucoma_seg_code/evaluate_jaccard_dice_institution.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder /projects/uthakuria@xsede.org/glaucoma_seg_datasets/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --cuda_num 0

echo "-- test job finished! --"