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
    parser = argparse.ArgumentParser(description="Run STAPLE algorithm on input images and save the result.")
    parser.add_argument("--prediction_folder", type=str, help="Path to the folder containing predictions")
    parser.add_argument("--label_folder", type=str, help="Path to the folder labels")
    parser.add_argument("--csv_path", type=str, help="Path to csv file with images, labels, and dataset names")
    parser.add_argument("--eval_disc", action='store_true', help="Whether to evaluate disc disc. Otherwise just evaluate cup")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    return parser.parse_args()


def image_pil_to_tensor(image_pil, eval_disc):
    # Load image as PIL image
    image = np.array(image_pil)
    
    # im = Image.fromarray(image)
    # im.save("/projects/uthakuria@xsede.org/image.jpeg")
    
    # Convert PIL image to tensor and flatten to 1D
    tensor = torch.tensor(image).view(-1, 3)
    
    # Define RGB to class mapping (adjust values based on your image)
    if eval_disc:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 1, (0, 0, 255): 1}
    else:
        class_mapping = {(255, 0, 0): 0, (0, 255, 0): 0, (0, 0, 255): 1}

    print(tensor.shape)
    
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


def main():
    args = parse_args()

    prediction_folder = args.prediction_folder
    label_folder = args.label_folder
    csv_path = args.csv_path
    eval_disc = args.eval_disc
    cuda_num = args.cuda_num
    results_info_folder = os.path.join(prediction_folder, 'results_info')
    os.makedirs(results_info_folder, exist_ok=True)
    results_info = os.path.join(results_info_folder, 'results_info.txt')
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)

    # Read CSV file
    df = pd.read_csv(csv_path)

    binrushed_metric = torchmetrics.classification.BinaryJaccardIndex()
    binrushed_dice = torchmetrics.classification.Dice(average='micro')

    chaksu_metric = torchmetrics.classification.BinaryJaccardIndex()
    chaksu_dice = torchmetrics.classification.Dice(average='micro')

    drishti_metric = torchmetrics.classification.BinaryJaccardIndex()
    drishti_dice = torchmetrics.classification.Dice(average='micro')

    g1020_metric = torchmetrics.classification.BinaryJaccardIndex()
    g1020_dice = torchmetrics.classification.Dice(average='micro')

    magrabi_metric = torchmetrics.classification.BinaryJaccardIndex()
    magrabi_dice = torchmetrics.classification.Dice(average='micro')

    messidor_metric = torchmetrics.classification.BinaryJaccardIndex()
    messidor_dice = torchmetrics.classification.Dice(average='micro')

    origa_metric = torchmetrics.classification.BinaryJaccardIndex()
    origa_dice = torchmetrics.classification.Dice(average='micro')

    refuge_metric = torchmetrics.classification.BinaryJaccardIndex()
    refuge_dice = torchmetrics.classification.Dice(average='micro')

    rimone_metric = torchmetrics.classification.BinaryJaccardIndex()
    rimone_dice = torchmetrics.classification.Dice(average='micro')

    riga_metric = torchmetrics.classification.BinaryJaccardIndex()
    riga_dice = torchmetrics.classification.Dice(average='micro')

    # Loop over rows in the CSV
    for index, row in df.iterrows():
        # Get the label path and construct the full paths for prediction and ground truth
        dataset_name = row["dataset_name"]
        if 'rimone' in dataset_name:
            dataset_name += '_dl' 
        label_path = os.path.join(label_folder + dataset_name + "/labels", os.path.basename(row['label_path']))
        prediction_path = os.path.join(prediction_folder, os.path.basename(row['image_path']))

        # Load prediction and ground truth images as PIL
        color_prediction = Image.open(prediction_path).convert('RGB')
        color_gt = grayscale_path_to_image_pil(label_path)

        # Convert to tensors with appropriate labeling (disc = disc+cup, cup = cup)
        prediction = image_pil_to_tensor(color_prediction, eval_disc).float()
        ground_truth = image_pil_to_tensor(color_gt, eval_disc)

        if row['dataset_name'] == 'binrushed':
            # Calculate Binary Jaccard Index
            binrushed_metric.update(prediction, ground_truth)
            binrushed_dice.update(prediction, ground_truth)
            print(f"Binrushed Image: {os.path.basename(label_path)}, Current Jaccard Index: {binrushed_metric.compute()}, current Dice: {binrushed_dice.compute()}")
        elif row['dataset_name'] == 'chaksu':
            # Calculate Binary Jaccard Index
            chaksu_metric.update(prediction, ground_truth)
            chaksu_dice.update(prediction, ground_truth)
            print(f"Chaksu Image: {os.path.basename(label_path)}, Current Jaccard Index: {chaksu_metric.compute()}, current Dice: {chaksu_dice.compute()}")
        elif row['dataset_name'] == 'drishti':
            # Calculate Binary Jaccard Index
            drishti_metric.update(prediction, ground_truth)
            drishti_dice.update(prediction, ground_truth)
            print(f"Drishti Image: {os.path.basename(label_path)}, Current Jaccard Index: {drishti_metric.compute()}, current Dice: {drishti_dice.compute()}")
        elif row['dataset_name'] == 'g1020':
            # Calculate Binary Jaccard Index
            g1020_metric.update(prediction, ground_truth)
            g1020_dice.update(prediction, ground_truth)
            print(f"G1020 Image: {os.path.basename(label_path)}, Current Jaccard Index: {g1020_metric.compute()}, current Dice: {g1020_dice.compute()}")
        elif row['dataset_name'] == 'magrabi':
            # Calculate Binary Jaccard Index
            magrabi_metric.update(prediction, ground_truth)
            magrabi_dice.update(prediction, ground_truth)
            print(f"Magrabi Image: {os.path.basename(label_path)}, Current Jaccard Index: {magrabi_metric.compute()}, current Dice: {magrabi_dice.compute()}")
        elif row['dataset_name'] == 'messidor':
            # Calculate Binary Jaccard Index
            messidor_metric.update(prediction, ground_truth)
            messidor_dice.update(prediction, ground_truth)
            print(f"Messidor Image: {os.path.basename(label_path)}, Current Jaccard Index: {messidor_metric.compute()}, current Dice: {messidor_dice.compute()}")
        elif row['dataset_name'] == 'origa':
            # Calculate Binary Jaccard Index
            origa_metric.update(prediction, ground_truth)
            origa_dice.update(prediction, ground_truth)
            print(f"Origa Image: {os.path.basename(label_path)}, Current Jaccard Index: {origa_metric.compute()}, current Dice: {origa_dice.compute()}")
        elif row['dataset_name'] == 'refuge':
            # Calculate Binary Jaccard Index
            refuge_metric.update(prediction, ground_truth)
            refuge_dice.update(prediction, ground_truth)
            print(f"Refuge Image: {os.path.basename(label_path)}, Current Jaccard Index: {refuge_metric.compute()}, current Dice: {refuge_dice.compute()}")
        elif row['dataset_name'] == 'rimone':
            # Calculate Binary Jaccard Index
            rimone_metric.update(prediction, ground_truth)
            rimone_dice.update(prediction, ground_truth)
            print(f"Rimone Image: {os.path.basename(label_path)}, Current Jaccard Index: {rimone_metric.compute()}, current Dice: {rimone_dice.compute()}")
        if row['dataset_name'] == 'messidor' or row['dataset_name'] == 'magrabi' or row['dataset_name'] == 'binrushed':
            # Calculate Binary Jaccard Index
            riga_metric.update(prediction, ground_truth)
            riga_dice.update(prediction, ground_truth)
            print(f"Riga Image: {os.path.basename(label_path)}, Current Jaccard Index: {riga_metric.compute()}, current Dice: {riga_dice.compute()}")

    with open(results_info, 'a') as f:
        if eval_disc:
            f.write(f"Evaluation of disc: \n")
        else:
            f.write(f"\nEvaluation of cup: \n")
        f.write(f"Binrushed Jaccard: {binrushed_metric.compute():.3f}, Dice: {binrushed_dice.compute():.3f}\n")
        f.write(f"Chaksu Jaccard: {chaksu_metric.compute():.3f}, Dice: {chaksu_dice.compute():.3f}\n")
        f.write(f"Drishti Jaccard: {drishti_metric.compute():.3f}, Dice: {drishti_dice.compute():.3f}\n")
        f.write(f"G1020 Jaccard: {g1020_metric.compute():.3f}, Dice: {g1020_dice.compute():.3f}\n")
        f.write(f"Magrabi Jaccard: {magrabi_metric.compute():.3f}, Dice: {magrabi_dice.compute():.3f}\n")
        f.write(f"Messidor Jaccard: {messidor_metric.compute():.3f}, Dice: {messidor_dice.compute():.3f}\n")
        f.write(f"Origa Jaccard: {origa_metric.compute():.3f}, Dice: {origa_dice.compute():.3f}\n")
        f.write(f"Refuge Jaccard: {refuge_metric.compute():.3f}, Dice: {refuge_dice.compute():.3f}\n")
        f.write(f"Rimone Jaccard: {rimone_metric.compute():.3f}, Dice: {rimone_dice.compute():.3f}\n")
        f.write(f"Riga Jaccard: {riga_metric.compute():.3f}, Dice: {riga_dice.compute():.3f}\n")
        
    # Calculate and print the average Jaccard Index
    print(f"Binrushed Jaccard: {binrushed_metric.compute():.3f}, Dice: {binrushed_dice.compute():.3f}")
    print(f"Chaksu Jaccard: {chaksu_metric.compute():.3f}, Dice: {chaksu_dice.compute():.3f}")
    print(f"Drishti Jaccard: {drishti_metric.compute():.3f}, Dice: {drishti_dice.compute():.3f}")
    print(f"G1020 Jaccard: {g1020_metric.compute():.3f}, Dice: {g1020_dice.compute():.3f}")
    print(f"Magrabi Jaccard: {magrabi_metric.compute():.3f}, Dice: {magrabi_dice.compute():.3f}")
    print(f"Messidor Jaccard: {messidor_metric.compute():.3f}, Dice: {messidor_dice.compute():.3f}")
    print(f"Origa Jaccard: {origa_metric.compute():.3f}, Dice: {origa_dice.compute():.3f}")
    print(f"Refuge Jaccard: {refuge_metric.compute():.3f}, Dice: {refuge_dice.compute():.3f}")
    print(f"Rimone Jaccard: {rimone_metric.compute():.3f}, Dice: {rimone_dice.compute():.3f}")
    print(f"Riga Jaccard: {riga_metric.compute():.3f}, Dice: {riga_dice.compute():.3f}")


if __name__ == "__main__":
    main()
