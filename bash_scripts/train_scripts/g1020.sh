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
python3 ../../fl_train_glaucoma_seg.py \
    --train_csv /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_train.csv \
    --val_csv /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs/train_outputs_g1020_v2 \
    --dataset_mean $combined_mean \
    --dataset_std $combined_std \
    --lr 0.00003 \
    --batch_size 16 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 100 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 \
    --cuda_num 1 \
    # --fl_finetuned /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs/train_outputs_binrushed_v2/model/best_global_model_epoch35.pth

# --train_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_train.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_train.csv
# --val_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_val.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_val.csv   