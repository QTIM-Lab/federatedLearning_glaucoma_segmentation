import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
import evaluate
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from ..datasets import ImageSegmentationDataset
from torchvision.transforms import functional as F
from ..utils import color_palette
import copy
import time
import transformers
import json
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline 3: Unweighted/Democratic Federated Averaging - All datasets get equal influence regardless of size")
    parser.add_argument("--train_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all train datasets")
    parser.add_argument("--val_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all validation datasets")
    parser.add_argument("--local_sites_training_epochs", type=int, default=10, help="Maximum number of epochs to train on each site (may stop early due to local early stopping)")
    parser.add_argument("--fl_rounds", type=int, default=100, help="Number of federated learning rounds")
    parser.add_argument("--csv_img_path_col", type=str, default='image_path', help="Column name in the csv for the path to the image")
    parser.add_argument("--csv_label_path_col", type=str, default='label_path', help="Column name in the csv for the path to the segmentation label")
    parser.add_argument("--output_directory", type=str, default='./outputs', help="Desired path for output files (model, val inferences, etc)")
    parser.add_argument('--dataset_mean', nargs='+', type=float, default=[0.768, 0.476, 0.289], help='Array of float values for mean')
    parser.add_argument('--dataset_std', nargs='+', type=float, default=[0.221, 0.198, 0.165], help='Array of float values for std')
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and testing")
    parser.add_argument('--jitters', nargs='+', type=float, default=[0.2, 0.2, 0.05, 0.05, 0.75], help='Array of float jitter values: brightness, contrast, saturation, hue, probability')
    parser.add_argument("--patience", type=int, default=7, help="Local early stopping patience (epochs without improvement before stopping local training)")
    parser.add_argument("--num_val_outputs_to_save", type=int, default=5, help="Number of examples from val to save.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    parser.add_argument("--fl_finetuned", type=str, default="", help="Path to the .pth file of the federated learning fine-tuned model")
    parser.add_argument("--fl_patience", type=int, default=5, help="Early stopping for federated learning rounds")
    parser.add_argument("--start_fl_round", type=int, default=0, help="Federated learning round to start from (useful for resuming)")
    return parser.parse_args()

# Function to load a dataset from a CSV file and make it a ImageSegmentationDataset Obj(len==dataset_size)
def load_dataset(csv_file, transform, csv_img_path_col, csv_label_path_col):
    data_df = pd.read_csv(csv_file)
    image_paths = data_df[csv_img_path_col].tolist()
    label_paths = data_df[csv_label_path_col].tolist()
    return ImageSegmentationDataset(image_paths, label_paths, transform=transform)

# Pipeline 3: Democratic/unweighted federated averaging - all datasets get equal voice
def FedAvg_Unweighted(models_weights, dataset_sizes=None):
    """
    Perform democratic unweighted federated averaging.
    Each dataset contributes equally (11.1% each) regardless of size.
    Promotes fairness: Magrabi (94 samples) has same influence as Chaksu (1,345 samples).
    
    Args:
        models_weights: List of model state dictionaries
        dataset_sizes: List of integers representing the number of samples in each dataset (unused in unweighted)
    
    Returns:
        averaged_weights: Democratically averaged model state dictionary with equal representation
    """
    num_models = len(models_weights)
    
    # Each model gets equal weight
    equal_weight = 1.0 / num_models
    
    print(f"Number of models: {num_models}")
    print(f"Equal weight per model: {equal_weight:.4f}")
    
    averaged_weights = {}
    
    # Iterating through each key in the state dictionary of the first model
    for key in models_weights[0].keys():
        # Get the original tensor to preserve its dtype and device
        original_tensor = models_weights[0][key]
        
        # Start with zeros with the same dtype and device as the original tensor
        averaged_sum = torch.zeros_like(original_tensor, dtype=torch.float32)
        
        # Add equal contributions from each model
        for i in range(num_models):
            averaged_sum += equal_weight * models_weights[i][key].float()
            
        # Convert back to original dtype and store the average
        averaged_weights[key] = averaged_sum.to(original_tensor.dtype)
    
    return averaged_weights

# Function to train one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device, batch_size):
    model.train()
    total_loss = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
        
        # Ensure class_labels are being extracted correctly
        if "class_labels" in batch:
            class_labels = [label.long().to(device) for label in batch["class_labels"]]
        else:
            raise ValueError("class_labels not found in batch")
        print("Training ongoing")
        optimizer.zero_grad()
        try:
            outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
        except RuntimeError as e:
            print(f"RuntimeError during forward pass: {e}")
            continue  

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (len(dataloader) * batch_size)

def validate_model(model, val_loader, preprocessor, criterion, device, metric, palette, inference_directory, epoch, num_val_outputs_to_save, id2label, site_name=""):
    model.eval()
    total_val_loss = 0.0
    total_samples = 0
    mean_ious = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            print(f"Val going for {site_name}")
            pixel_values = batch["pixel_values"].to(device)
            # Forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]])
            # Loss calculation
            loss = outputs.loss
            total_val_loss += loss.item()
            # Post-processing for metric calculation
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            # Metric update
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]
            metric.add_batch(predictions=predicted_segmentation_maps, references=ground_truth_segmentation_maps)
            # Visualization and saving
            if idx < num_val_outputs_to_save:
                for i, (original_img, predicted_map, ground_truth_map) in enumerate(zip(original_images, predicted_segmentation_maps, batch["original_segmentation_maps"])):
                    # For predicted segmentation map
                    pred_color_seg_map = np.zeros_like(original_img)
                    for label_id, color in enumerate(palette):
                        pred_color_seg_map[predicted_map.cpu() == label_id, :] = color
                    blended_pred_image = (original_img * 0.7 + pred_color_seg_map * 0.3).astype(np.uint8)

                    # For ground truth segmentation map
                    gt_color_seg_map = np.zeros_like(original_img)
                    for label_id, color in enumerate(palette):
                        gt_color_seg_map[ground_truth_map == label_id, :] = color
                    blended_gt_image = (original_img * 0.7 + gt_color_seg_map * 0.3).astype(np.uint8)

                    # Save both images
                    plt.figure(figsize=(10, 10))
                    plt.imshow(blended_pred_image)
                    plt.axis('off')
                    plt.savefig(f"{inference_directory}/round_{epoch}_{site_name}_batch_{idx}_image_{i}_val_segmentation.png", bbox_inches='tight')
                    plt.close()

                    plt.figure(figsize=(10, 10))
                    plt.imshow(blended_gt_image)
                    plt.axis('off')
                    plt.savefig(f"{inference_directory}/round_{epoch}_{site_name}_batch_{idx}_image_{i}_gt_segmentation.png", bbox_inches='tight')
                    plt.close()
            total_samples += len(pixel_values)

        mean_iou = metric.compute(num_labels=len(id2label), ignore_index=0)["mean_iou"]
        mean_ious.append(mean_iou)
        print(f"Mean IoU for {site_name}:", mean_iou)

    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0
    avg_mean_iou = sum(mean_ious) / len(mean_ious) if mean_ious else 0
    return avg_val_loss, avg_mean_iou

def find_latest_checkpoint(model_directory):
    pattern = re.compile(r"best_global_model_round_(\d+)_metric_.*\.pth")
    max_round_num = -1
    latest_checkpoint_path = None

    for filename in os.listdir(model_directory):
        match = pattern.match(filename)
        if match:
            round_num = int(match.group(1))
            if round_num > max_round_num:
                max_round_num = round_num
                latest_checkpoint_path = os.path.join(model_directory, filename)

    return max_round_num, latest_checkpoint_path


def main():
    start_time = time.time()
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    csv_img_path_col = args.csv_img_path_col
    csv_label_path_col = args.csv_label_path_col
    output_directory = args.output_directory
    dataset_mean = args.dataset_mean
    dataset_std = args.dataset_std
    lr = args.lr
    batch_size = args.batch_size
    jitters = args.jitters
    patience = args.patience
    num_val_outputs_to_save = args.num_val_outputs_to_save
    num_workers = args.num_workers
    cuda_num = args.cuda_num
    local_sites_training_epochs = args.local_sites_training_epochs
    fl_patience = args.fl_patience
    fl_rounds = args.fl_rounds
    
    assert len(jitters) == 5, 'jitters must have 5 values'
    assert (jitters[0] < 1 and jitters[0] > 0 
            and jitters[1] <= 1 and jitters[1] >= 0 
            and jitters[2] <= 1 and jitters[2] >= 0 
            and jitters[3] <= 1 and jitters[3] >= 0
            and jitters[4] <= 1 and jitters[4] >= 0), 'jitters must be [0,1]'
    assert len(dataset_mean) == 3, 'dataset mean must have 3 float values'
    assert len(dataset_std) == 3, 'dataset std must have 3 float values'
    
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)
            
    # Initialize output folders (use repo-local paths)
    os.makedirs(args.output_directory, exist_ok=True)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_name = os.path.basename(output_directory)
    model_directory = os.path.join(base_dir, "models", project_name)
    os.makedirs(model_directory, exist_ok=True)
    inference_directory = os.path.join(base_dir, 'outputs', project_name)
    os.makedirs(inference_directory, exist_ok=True)
    dataset_names = [os.path.basename(csv_file).replace('_train.csv', '') for csv_file in args.train_csv]
    
    # Save args as JSON
    args_dict = vars(args)  # Convert args namespace to dict
    args_dict['train_py_file'] = os.path.basename(__file__) 
    with open(os.path.join(args.output_directory, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)
    
    # for classes
    id2label = {
        0: "unlabeled",
        1: "bg",
        2: "disc",
        3: "cup"
    }
    
    # for vis
    palette = color_palette()

    # transforms
    ADE_MEAN = np.array(dataset_mean)
    ADE_STD = np.array(dataset_std)

    # Transforms
    train_transform = A.Compose([
        A.ColorJitter(brightness=args.jitters[0], contrast=args.jitters[1], saturation=args.jitters[2], hue=args.jitters[3], p=args.jitters[4]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=args.dataset_mean, std=args.dataset_std),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])
    
    # Load custom datasets for each CSV file, list of ImageSegmentationDataset Objs()
    train_datasets = [load_dataset(csv_file, train_transform, args.csv_img_path_col, args.csv_label_path_col) for csv_file in args.train_csv]
    val_datasets = [load_dataset(csv_file, val_transform, args.csv_img_path_col, args.csv_label_path_col) for csv_file in args.val_csv]
    
    # Calculate dataset sizes for weighted averaging
    dataset_sizes = [len(dataset) for dataset in train_datasets]
    print(f"Training dataset sizes: {dict(zip(dataset_names, dataset_sizes))}")
    
    # Create an empty preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    def collate_fn(batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]

        # Apply the preprocessor
        processed_batch = preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )

        # Check if processed_batch is a BatchFeature object and convert to dict if necessary
        if isinstance(processed_batch, transformers.image_processing_utils.BatchFeature):
            processed_batch = dict(processed_batch)

        # Add original images and segmentation maps to the batch
        processed_batch["original_images"] = inputs[2]
        processed_batch["original_segmentation_maps"] = inputs[3]
        
        return processed_batch
    

    # Data loaders for each dataset
    train_dataloaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn) for ds in train_datasets]
    val_dataloaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn) for ds in val_datasets]
    
    # Experiment ID for metrics
    experiment_id = output_directory.split('/')[-1] + str(time.time())
    metric = evaluate.load("mean_iou", experiment_id=experiment_id)
    # print(f'begin train, len train: {len(train_dataloaders[0])}, len val: {len(val_dataloaders[0])}')

    
    def log_metrics(fl_round, train_losses, global_val_losses, global_mean_ious, dataset_sizes, equal_weights):
        """Logs aggregated metrics to a text file with unweighted fedavg information."""
        metrics_file_path = os.path.join(args.output_directory, 'fl_round_metrics.txt')
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(global_val_losses) / len(global_val_losses)
        avg_mean_iou = sum(global_mean_ious) / len(global_mean_ious)
        
        with open(metrics_file_path, 'a') as f:
            f.write(f"FL Round: {fl_round+1}, Avg Train Loss: {avg_train_loss:.4f}, Avg Global Val Loss: {avg_val_loss:.4f}, Avg Global Mean IoU: {avg_mean_iou:.4f}\n")
            f.write(f"  Dataset sizes: {dataset_sizes}\n")
            f.write(f"  Equal weights used: {[f'{w:.4f}' for w in equal_weights]}\n")
            f.write(f"  Individual site train losses: {[f'{loss:.4f}' for loss in train_losses]}\n")
            f.write(f"  Individual site global val losses: {[f'{loss:.4f}' for loss in global_val_losses]}\n")
            f.write(f"  Individual site global mean IoUs: {[f'{iou:.4f}' for iou in global_mean_ious]}\n")

    def plot_loss_curves(fl_round, round_train_losses, round_val_losses):
        """Updates and saves loss curves after each FL round."""
        plt.figure(figsize=(15, 5))
        
        # Plot training losses by site
        plt.subplot(1, 2, 1)
        for i, site_name in enumerate(dataset_names):
            site_train_losses = [round_train_losses[round][i] for round in range(len(round_train_losses))]
            plt.plot(site_train_losses, label=f'{site_name}', marker='o')
        plt.title('Training Loss by FL Round and Site (Unweighted FedAvg)')
        plt.xlabel('FL Round')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot global validation losses by site
        plt.subplot(1, 2, 2)
        for i, site_name in enumerate(dataset_names):
            site_val_losses = [round_val_losses[round][i] for round in range(len(round_val_losses))]
            plt.plot(site_val_losses, label=f'{site_name}', marker='s')
        plt.title('Global Model Validation Loss by FL Round and Site (Unweighted FedAvg)')
        plt.xlabel('FL Round')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_directory, f'fl_loss_curves_round_{fl_round+1}.png'))
        plt.close()
        
    def save_model(model, model_directory, fl_round, metric):
        """Saves the global model checkpoint."""
        model_path = os.path.join(model_directory, f'best_global_model_round_{fl_round+1}_metric_{metric:.4f}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def global_early_stopping_check(current_metric, best_metric, counter, patience):
        updated = False
        if current_metric < best_metric:
            best_metric = current_metric
            counter = 0
            updated = True
            print(f"New best global model saved with metric: {current_metric}")
        else:
            counter += 1
            if counter >= patience:
                print("Global early stopping triggered.")
                return True, best_metric, counter, updated
        return False, best_metric, counter, updated

    global_weights = None
    global_best_val_metric = float('inf')
    early_stop_counter = 0
    
    # Track losses across rounds for plotting
    round_train_losses = []  # List of lists: [round][site_index]
    round_val_losses = []    # List of lists: [round][site_index]
    
    # Determine the FL round to resume from and get the latest checkpoint path
    start_fl_round, latest_checkpoint_path = find_latest_checkpoint(model_directory)

    # Load the checkpoint if it exists before entering the FL rounds loop
    if latest_checkpoint_path:
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        global_weights = checkpoint
        print(f"Resuming from FL round {start_fl_round + 1} using checkpoint: {latest_checkpoint_path}")
    else:
        print("No checkpoint found. Starting from the beginning.")
        start_fl_round = 0
    
    # fl_round loop
    for fl_round in range(start_fl_round, fl_rounds):
        print(f"Federated Learning Round {fl_round+1}/{fl_rounds}")
        model_states = []
        round_train_loss_per_site = []

        # Phase 1: Local training on each site
        for site_index in range(len(train_dataloaders)):
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                        id2label=id2label, 
                                                                        ignore_mismatched_sizes=True).to(device)
            # If a fine-tuning checkpoint is specified, load it
            if args.fl_finetuned:
                checkpoint = torch.load(args.fl_finetuned, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"Loaded the FL-finetuned weights from {args.fl_finetuned}")
            elif global_weights:
                # Or if a global model state exists from previous rounds, load it
                model.load_state_dict(global_weights)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            criterion = torch.nn.CrossEntropyLoss()
            
            train_loader = train_dataloaders[site_index]
            site_name = dataset_names[site_index]
            
            # Local training with early stopping
            site_train_losses = []
            site_val_losses = []
            best_local_val_loss = float('inf')
            local_patience_counter = 0
            best_local_model_state = None
            
            val_loader = val_dataloaders[site_index]
            
            for epoch in range(local_sites_training_epochs):
                print(f"Site {site_name}, Epoch {epoch+1}/{local_sites_training_epochs}")
                
                # Training
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, batch_size)
                site_train_losses.append(train_loss)
                
                # Local validation for early stopping
                local_val_loss, local_mean_iou = validate_model(model, val_loader, preprocessor, 
                                                               criterion, device, metric, 
                                                               palette, inference_directory, 
                                                               f"{fl_round}_local_{epoch}", 
                                                               0, id2label, f"{site_name}_local")
                site_val_losses.append(local_val_loss)
                
                print(f"Site {site_name}, Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Val Loss: {local_val_loss:.4f}")
                
                # Local early stopping check
                if local_val_loss < best_local_val_loss:
                    best_local_val_loss = local_val_loss
                    local_patience_counter = 0
                    best_local_model_state = copy.deepcopy(model.state_dict())
                    print(f"Site {site_name}: New best local model at epoch {epoch+1}")
                else:
                    local_patience_counter += 1
                    print(f"Site {site_name}: No improvement for {local_patience_counter} epochs")
                
                # Check if early stopping should trigger
                if local_patience_counter >= patience:
                    print(f"Site {site_name}: Local early stopping triggered at epoch {epoch+1}")
                    break
                    
                scheduler.step()
            
            # Use the best local model if early stopping occurred, otherwise use the final model
            if best_local_model_state is not None:
                final_model_state = best_local_model_state
                final_val_loss = best_local_val_loss
                print(f"Site {site_name}: Using best local model with val loss {final_val_loss:.4f}")
            else:
                final_model_state = model.state_dict()
                final_val_loss = site_val_losses[-1] if site_val_losses else float('inf')
                print(f"Site {site_name}: Using final model with val loss {final_val_loss:.4f}")
            
            # Store metrics and model state
            avg_site_train_loss = sum(site_train_losses) / len(site_train_losses)
            round_train_loss_per_site.append(avg_site_train_loss)
            model_states.append(copy.deepcopy(final_model_state))
            
            # Cleanup
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            
        # Phase 2: Pipeline 3 - Democratic unweighted federated averaging (all sites equal)
        print("Performing Pipeline 3: Democratic unweighted federated averaging...")
        global_weights = FedAvg_Unweighted(model_states, dataset_sizes)
        
        # Calculate equal weights for logging
        num_models = len(model_states)
        current_equal_weights = [1.0 / num_models] * num_models
        
        # Phase 3: Global model evaluation on each site
        print("Evaluating global model on all sites...")
        global_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                            id2label=id2label, 
                                                                            ignore_mismatched_sizes=True).to(device)
        global_model.load_state_dict(global_weights)
        
        round_val_loss_per_site = []
        round_mean_iou_per_site = []
        
        for site_index in range(len(val_dataloaders)):
            val_loader = val_dataloaders[site_index]
            site_name = dataset_names[site_index]
            
            val_loss, mean_iou = validate_model(global_model, val_loader, preprocessor, 
                                               torch.nn.CrossEntropyLoss(), device, metric, 
                                               palette, inference_directory, fl_round, 
                                               num_val_outputs_to_save, id2label, site_name)
            
            round_val_loss_per_site.append(val_loss)
            round_mean_iou_per_site.append(mean_iou)
            
            print(f"Global model on {site_name}: Val Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
        
        # Store losses for plotting
        round_train_losses.append(round_train_loss_per_site)
        round_val_losses.append(round_val_loss_per_site)
        
        # Compute the mean validation metric across sites for the current round
        mean_val_metric = sum(round_val_loss_per_site) / len(round_val_loss_per_site)
        
        # Log metrics with unweighted fedavg information
        log_metrics(fl_round, round_train_loss_per_site, round_val_loss_per_site, round_mean_iou_per_site, dataset_sizes, current_equal_weights)
        
        # Plot loss curves
        plot_loss_curves(fl_round, round_train_losses, round_val_losses)
        
        # Check for global early stopping
        stop, global_best_val_metric, early_stop_counter, updated = global_early_stopping_check(
            mean_val_metric, global_best_val_metric, early_stop_counter, args.fl_patience)
        
        if updated:
            # Save the updated global model
            save_model(global_model, model_directory, fl_round, global_best_val_metric)
        
        if stop:
            print(f"Global early stopping triggered after {fl_round+1} rounds.")
            break
            
        # Cleanup global model
        del global_model
        torch.cuda.empty_cache()

        print(f"Federated Learning Round {fl_round+1} completed.")
        print(f"Average global validation loss: {mean_val_metric:.4f}")
        print(f"Average global mean IoU: {sum(round_mean_iou_per_site)/len(round_mean_iou_per_site):.4f}")

    # Save final summary with unweighted fedavg information
    with open(os.path.join(args.output_directory, 'training_summary.txt'), 'w') as f:
        f.write("=== FEDERATED LEARNING WITH GLOBAL EVALUATION AND UNWEIGHTED FEDAVG SUMMARY ===\n")
        f.write(f"Training datasets: {dataset_names}\n")
        f.write(f"Training dataset sizes: {dict(zip(dataset_names, dataset_sizes))}\n")
        f.write("Note: Standard unweighted federated averaging used - all models contribute equally\n")
        f.write("Note: Local early stopping used for each site's training\n")
        f.write(f"FL rounds completed: {fl_round+1}\n")
        f.write(f"Max local training epochs per round: {local_sites_training_epochs}\n")
        f.write(f"Best global validation loss: {global_best_val_metric:.4f}\n")
        f.write(f"Early stopping: {'Yes' if stop else 'No'}\n")
        f.write("Note: Global model evaluated on each site's validation set after federated averaging\n")

if __name__ == "__main__":
    main() 