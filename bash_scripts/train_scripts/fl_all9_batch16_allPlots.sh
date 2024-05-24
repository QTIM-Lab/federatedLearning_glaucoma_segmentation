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

# train
python3 ../../fl_train_glaucoma_seg.py \
    --train_csv /sddata/data/retina_datasets_preprocessed/output_csvs/binrushed_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/chaksu_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/drishti_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/magrabi_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/origa_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/refuge_train.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/rimone_train.csv \
    --val_csv /sddata/data/retina_datasets_preprocessed/output_csvs/binrushed_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/chaksu_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/drishti_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/magrabi_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/origa_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/refuge_val.csv \
                /sddata/data/retina_datasets_preprocessed/output_csvs/rimone_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs/fl_all9_batch16_allPlots \
    --dataset_mean $combined_mean \
    --dataset_std $combined_mean \
    --lr 0.00002 \
    --batch_size 16 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 100 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 \
    --cuda_num 0
    # --fl_finetuned /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train__outputs_binDriOriga/model/best_global_model_epoch31.pth
    # --fl_finetuned /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_RIGA/model/best_global_model_epoch62.pth

# --train_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_train.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_train.csv
# --val_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_val.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_val.csv   