import argparse
from PIL import Image
import numpy as np
import torch
import torchmetrics
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate per-sample dice scores and save to CSV.")
    parser.add_argument("--prediction_folder", type=str, help="Path to the folder containing predictions")
    parser.add_argument("--label_folder", type=str, help="Path to the folder labels")
    parser.add_argument("--csv_path", type=str, help="Path to csv file with images, labels, and dataset names")
    parser.add_argument("--eval_disc", action='store_true', help="Whether to evaluate disc disc. Otherwise just evaluate cup")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    parser.add_argument("--output_csv", type=str, help="Path to save per-sample results CSV")
    parser.add_argument("--model_name", type=str, help="Name of the model being evaluated (for statistical analysis)")
    parser.add_argument("--statistical_output_dir", type=str, help="Root directory for statistical analysis CSVs (e.g., 'scores')")
    return parser.parse_args()


def image_pil_to_tensor(image_pil, eval_disc):
    # Load image as PIL image
    image = np.array(image_pil)
    
    # Convert PIL image to tensor and flatten to 1D
    tensor = torch.tensor(image).view(-1, 3)
    
    # Define RGB to class mapping (adjust values based on your image)
    if eval_disc:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 1, (0, 0, 255): 1}
    else:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 0, (0, 0, 255): 1}
    
    # Map RGB values to class labels
    class_labels = torch.tensor([class_mapping[tuple(rgb.tolist())] for rgb in tensor])
    
    # Reshape to 2D
    class_labels = class_labels.view(image.shape[1], image.shape[0])
    
    return class_labels


def grayscale_path_to_image_pil(image_path):
    # Load image as PIL image
    gray_image = np.array(Image.open(image_path).convert('L'))
    # Create a 3-channel image with the same shape as the grayscale image
    image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

    # Assign red, green, and blue colors based on intensity values
    image[gray_image == 0] = [255, 0, 0]   # Values that were 0 become red
    image[gray_image == 127] = [0, 255, 0]  # Values that were 127 become green
    image[gray_image == 255] = [0, 0, 255]  # Values that were 255 become blue

    return Image.fromarray(image)


def calculate_dice_score(prediction, ground_truth, device):
    """Calculate dice score for a single sample"""
    dice_metric = torchmetrics.classification.Dice(average='micro').to(device)
    # Move tensors to device
    prediction = prediction.to(device)
    ground_truth = ground_truth.to(device)
    dice_score = dice_metric(prediction, ground_truth)
    return dice_score.cpu().item()


def calculate_jaccard_score(prediction, ground_truth, device):
    """Calculate jaccard score for a single sample"""
    jaccard_metric = torchmetrics.classification.BinaryJaccardIndex().to(device)
    # Move tensors to device
    prediction = prediction.to(device)
    ground_truth = ground_truth.to(device)
    jaccard_score = jaccard_metric(prediction, ground_truth)
    return jaccard_score.cpu().item()


def save_for_statistical_analysis(per_sample_results, model_name, eval_disc, statistical_output_dir):
    """
    Save results in format required by statistical_analysis.py
    
    Creates/appends to dataset-specific CSV files in scores/disc/ or scores/cup/
    Format: image_name, model_name, dice_score
    
    Args:
        per_sample_results: List of dicts with per-sample results
        model_name: Name of the model being evaluated
        eval_disc: Whether evaluating disc (True) or cup (False)
        statistical_output_dir: Root directory (e.g., 'scores')
    """
    if not model_name or not statistical_output_dir:
        return
    
    # Determine subdirectory based on eval_disc
    eval_type = 'disc' if eval_disc else 'cup'
    output_subdir = os.path.join(statistical_output_dir, eval_type)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Group results by dataset
    dataset_results = {}
    for result in per_sample_results:
        dataset_name = result['dataset_name']
        if dataset_name not in dataset_results:
            dataset_results[dataset_name] = []
        
        # Extract image name from path (without extension)
        image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        
        dataset_results[dataset_name].append({
            'image_name': image_name,
            'model_name': model_name,
            'dice_score': result['dice_score']
        })
    
    # Save each dataset to its own CSV file
    for dataset_name, results in dataset_results.items():
        csv_filename = f"{dataset_name}.csv"
        csv_path = os.path.join(output_subdir, csv_filename)
        
        # Create DataFrame for new results
        new_df = pd.DataFrame(results)
        
        # Check if file exists and append/merge with existing data
        if os.path.exists(csv_path):
            # Read existing data
            existing_df = pd.read_csv(csv_path)
            
            # Remove any existing entries for this model (to avoid duplicates)
            existing_df = existing_df[existing_df['model_name'] != model_name]
            
            # Concatenate with new results
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"Updated statistical analysis CSV: {csv_path}")
        else:
            # Create new file
            new_df.to_csv(csv_path, index=False)
            print(f"Created statistical analysis CSV: {csv_path}")
    
    print(f"\nStatistical analysis CSVs saved to: {output_subdir}/")
    print(f"   Format: image_name, model_name, dice_score")
    print(f"   Datasets: {', '.join(dataset_results.keys())}")


def main():
    args = parse_args()

    prediction_folder = args.prediction_folder
    label_folder = args.label_folder
    csv_path = args.csv_path
    eval_disc = args.eval_disc
    cuda_num = args.cuda_num
    output_csv = args.output_csv
    model_name = args.model_name
    statistical_output_dir = args.statistical_output_dir
    
    # Create results folder if needed
    results_info_folder = os.path.join(prediction_folder, 'results_info')
    os.makedirs(results_info_folder, exist_ok=True)
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # List to store per-sample results
    per_sample_results = []

    # Aggregated metrics per dataset (like original script)
    dataset_metrics = {}
    dataset_dice = {}
    
    # Initialize metrics for each unique dataset
    unique_datasets = df['dataset_name'].unique()
    for dataset in unique_datasets:
        dataset_metrics[dataset] = torchmetrics.classification.BinaryJaccardIndex().to(device)
        dataset_dice[dataset] = torchmetrics.classification.Dice(average='micro').to(device)

    print(f"Processing {len(df)} samples...")

    # Loop over rows in the CSV
    for index, row in df.iterrows():
        # Get the label path and construct the full paths for prediction and ground truth
        dataset_name = row["dataset_name"]
        original_dataset_name = dataset_name
        
        # Use the full label path from CSV (no reconstruction needed)
        label_path = row['label_path']
        prediction_path = os.path.join(prediction_folder, os.path.basename(row['image_path']))

        # Check if files exist
        if not os.path.exists(prediction_path):
            print(f"Warning: Prediction file not found: {prediction_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            continue

        try:
            # Load prediction and ground truth images as PIL
            color_prediction = Image.open(prediction_path).convert('RGB')
            color_gt = grayscale_path_to_image_pil(label_path)

            # Convert to tensors with appropriate labeling (disc = disc+cup, cup = cup)
            prediction = image_pil_to_tensor(color_prediction, eval_disc).float()
            ground_truth = image_pil_to_tensor(color_gt, eval_disc)

            # Calculate per-sample scores
            sample_dice = calculate_dice_score(prediction, ground_truth, device)
            sample_jaccard = calculate_jaccard_score(prediction, ground_truth, device)

            # Store per-sample result
            sample_result = {
                'image_path': row['image_path'],
                'label_path': row['label_path'],
                'dataset_name': original_dataset_name,
                'dice_score': sample_dice,
                'jaccard_score': sample_jaccard,
                'eval_type': 'disc' if eval_disc else 'cup'
            }
            per_sample_results.append(sample_result)

            # Update aggregated metrics (like original script)
            if original_dataset_name in dataset_metrics:
                # Move tensors to device for aggregated metrics
                pred_device = prediction.to(device)
                gt_device = ground_truth.to(device)
                dataset_metrics[original_dataset_name].update(pred_device, gt_device)
                dataset_dice[original_dataset_name].update(pred_device, gt_device)

            if index % 50 == 0:
                print(f"Processed {index+1}/{len(df)} samples...")

        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            continue

    # Save per-sample results to CSV
    if output_csv:
        results_df = pd.DataFrame(per_sample_results)
        results_df.to_csv(output_csv, index=False)
        print(f"Per-sample results saved to: {output_csv}")
    
    # Save results for statistical analysis (if model_name and statistical_output_dir provided)
    if model_name and statistical_output_dir:
        save_for_statistical_analysis(per_sample_results, model_name, eval_disc, statistical_output_dir)
    
    # Print and save aggregated results (like original script)
    results_info = os.path.join(results_info_folder, 'results_info.txt')
    
    with open(results_info, 'a') as f:
        if eval_disc:
            f.write(f"Evaluation of disc: \n")
        else:
            f.write(f"\nEvaluation of cup: \n")
        
        print(f"\n{'='*50}")
        print(f"AGGREGATED RESULTS ({'DISC' if eval_disc else 'CUP'})")
        print(f"{'='*50}")
        
        for dataset in unique_datasets:
            if dataset in dataset_metrics:
                jaccard_score = dataset_metrics[dataset].compute()
                dice_score = dataset_dice[dataset].compute()
                
                f.write(f"{dataset} Jaccard: {jaccard_score:.3f}, Dice: {dice_score:.3f}\n")
                print(f"{dataset} Jaccard: {jaccard_score:.3f}, Dice: {dice_score:.3f}")

    print(f"\nAggregated results saved to: {results_info}")
    print(f"Total samples processed: {len(per_sample_results)}")


if __name__ == "__main__":
    main() 