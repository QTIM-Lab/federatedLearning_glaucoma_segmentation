import pandas as pd
import numpy as np

csv_path = '/home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_binrushed_alone/results.csv'
df = pd.read_csv(csv_path)

# computing the overall metrics
mean_jaccard = df['jaccard_scores'].mean()
median_jaccard = df['jaccard_scores'].median()
mean_dice = df['dice_scores'].mean()
median_dice = df['dice_scores'].median()

print("Overall Metrics:")
print(f"Mean Jaccard: {mean_jaccard}")
print(f"Median Jaccard: {median_jaccard}")
print(f"Mean Dice: {mean_dice}")
print(f"Median Dice: {median_dice}")

# computing the dataset-specific metrics
datasets = df['dataset_name'].unique()
for dataset in datasets:
    dataset_df = df[df['dataset_name'] == dataset]
    mean_jaccard = dataset_df['jaccard_scores'].mean()
    median_jaccard = dataset_df['jaccard_scores'].median()
    mean_dice = dataset_df['dice_scores'].mean()
    median_dice = dataset_df['dice_scores'].median()
    
    print(f"\n{dataset} Metrics:")
    print(f"Mean Jaccard: {mean_jaccard}")
    print(f"Median Jaccard: {median_jaccard}")
    print(f"Mean Dice: {mean_dice}")
    print(f"Median Dice: {median_dice}")