import argparse
from PIL import Image
import os
import torch
import torch.multiprocessing as mp
from multiprocessing import Queue, Manager
from torchvision import transforms
import csv
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
import albumentations as A

from utils import color_palette


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--model_path", type=str, default='/path/to/weights.pt', help="Path to trained model weights")
    parser.add_argument("--input_csv", type=str, default='/path/for/images.csv', help="Path to csv to run inference on all rows")
    parser.add_argument("--csv_path_col_name", required=True, default='img_path', help="Column name for the paths to images in the csv, ex: file_path or img_path")
    parser.add_argument("--output_root_dir", type=str, required=True, default='./', help="Path to outputs for CSV and images. CSV will be saved to root dir, images to root_dir/outputs")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of threads to run in parallel")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    return parser.parse_args()

# Function to find maximum diameter and draw the line
def find_and_draw_max_diameter(contour, image, color_tuple):
    max_diameter = 0
    max_diameter_points = None

    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            x1, y1 = contour[i][0]
            x2, y2 = contour[j][0]

            if abs(x1 - x2) == 0:  # Check if the two points are on the same vertical line
                dist = abs(y1 - y2)  # Vertical distance
                if dist > max_diameter:
                    max_diameter = dist
                    max_diameter_points = ((x1, y1), (x2, y2))
    
    if max_diameter_points is not None:
        try:
            cv2.line(image, max_diameter_points[0], max_diameter_points[1], color_tuple, 2)
        except Exception as e:
            # Print the error.
            print(f'Error: {e}')

    return max_diameter

def inference_function(model, row, result_queue, idx, transform, preprocessor, palette, output_dir, csv_path_col_name, device):
    print("running inference")

    img_pil = Image.open(row[csv_path_col_name]).convert('RGB')
    img_pil_copy = np.array(img_pil.copy())
    width, height = img_pil.size
    transformed_image = transform(image=np.array(img_pil))['image'].transpose(2,0,1)

    # prepare image for the model
    inputs = preprocessor(images=transformed_image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [(height, width)]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                target_sizes=target_sizes)
    
    # Val
    pred_segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

    color_segmentation_map = np.zeros((pred_segmentation_map.shape[0], pred_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_segmentation_map[pred_segmentation_map - 1 == label, :] = color
    # Convert to BGR
    pred_color_seg = color_segmentation_map[..., ::1]

    pred_overlay = np.array(img_pil) * 0.7 + pred_color_seg * 0.3
    pred_overlay = pred_overlay.astype(np.uint8)

    # Combine the channels (you can use any operation that suits your needs)
    total_disk_seg = pred_color_seg[:,:,1] + pred_color_seg[:,:,2]

    # Clip values to be within the range [0, 255]
    total_disk_seg = np.clip(total_disk_seg, 0, 255)

    disk_contours, _ = cv2.findContours(total_disk_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(disk_contours) > 0:
        disk_contours = max(disk_contours, key=cv2.contourArea)

    cup_contours, _ = cv2.findContours(pred_color_seg[:,:,2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cup_contours) > 0:
        cup_contours = max(cup_contours, key=cv2.contourArea)

    cup_diameter = 0.0
    disk_diameter = 0.0

    
    # Disk convexity and circularity
    if len(disk_contours) > 0:
        disk_contour_area = cv2.contourArea(disk_contours)
        disk_perimeter = cv2.arcLength(disk_contours, True)
        if disk_perimeter == 0:  # Avoid division by zero
            disk_circularity = 0
        else:
            disk_circularity = (4 * np.pi * disk_contour_area) / (disk_perimeter ** 2)
        disk_convex_hull = cv2.convexHull(disk_contours)
        disk_convex_hull_area = cv2.contourArea(disk_convex_hull)
        if disk_convex_hull_area > 0:
            disk_convexity = disk_contour_area / disk_convex_hull_area
        # Disk Diameter
        try:
            cv2.drawContours(img_pil_copy, [disk_contours], -1, (0, 255, 0), 4)  # Green color for disk_contours
            disk_diameter = find_and_draw_max_diameter(disk_contours, img_pil_copy, (0,255,0))
        except:
            print("disk dia error on: ", idx + 1)
    # Cup convexity
    if len(cup_contours) > 0:
        cup_contour_area = cv2.contourArea(cup_contours)
        cup_perimeter = cv2.arcLength(cup_contours, True)
        if cup_perimeter == 0:  # Avoid division by zero
            cup_circularity = 0
        else:
            cup_circularity = (4 * np.pi * cup_contour_area) / (cup_perimeter ** 2)
        cup_convex_hull = cv2.convexHull(cup_contours)
        cup_convex_hull_area = cv2.contourArea(cup_convex_hull)
        if cup_convex_hull_area:
            cup_convexity = cup_contour_area / cup_convex_hull_area
        # Cup Diameter
        try:
            cv2.drawContours(img_pil_copy, [cup_contours], -1, (0, 0, 255), 4)  # Green color for disk_contours
            cup_diameter = find_and_draw_max_diameter(cup_contours, img_pil_copy, (0,0,255))
        except:
            print("cup dia error on: ", idx + 1)

    cdr = 0
    if disk_diameter > 0:
        cdr = cup_diameter / disk_diameter

    # composite = np.concatenate((np.array(img_pil), img_pil_copy), axis=1)

    row_path_name = row[csv_path_col_name].split('/')[-1]

    # plt.imshow(composite)
    # plt.imshow(pred_color_seg)
    # plt.axis('off')
    # plt.title(f'Img{idx + 1}')
    # plt.suptitle(f'Composite')
    # composite_filename = os.path.join(output_dir, f'composite_{row_path_name}')
    # plt.savefig(composite_filename, bbox_inches='tight')
    # plt.close()

    # Convert the NumPy array to an image
    seg_output = Image.fromarray(pred_color_seg.astype('uint8'))

    # Save the image
    seg_output.save(os.path.join(output_dir, f'{row_path_name}'))

    row['pred_cdr'] = cdr

    result_queue.put(row)

    print(f"processed idx: {idx}")

def main():
    args = parse_args()
    model_path = args.model_path
    input_csv = args.input_csv
    csv_path_col_name = args.csv_path_col_name
    output_root_dir = args.output_root_dir
    num_processes = args.num_processes
    cuda_num = args.cuda_num

    # Create the output dir for the images
    output_dir = os.path.join(output_root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Multiprocess set to spawn. You can change depending on environment
    mp.set_start_method('spawn', force=True)

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)

    # for classes
    id2label = {
        0: "unlabeled",
        1: "background",
        2: "disc",
        3: "cup"
    }

    # for vis
    palette = color_palette()

    # transforms. use what you calculated for train/val
    ADE_MEAN = np.array([0.709, 0.439, 0.287])
    ADE_STD = np.array([0.210, 0.220, 0.199])

    test_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])

    # Replace the head of the pre-trained model
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                id2label=id2label, 
                                                                ignore_mismatched_sizes=True).to(device)
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    # Specify output file path to be hard-coded as results.csv in the output dir
    output_csv_file = os.path.join(output_root_dir, f'results.csv')

    # Create a multiprocessing queue to store processed rows
    result_queue = mp.Manager().Queue()

    # Multiprocessing
    with mp.Pool(num_processes) as pool:
        # Read the input CSV file
        with open(input_csv, 'r') as csv_file_reader:
            reader = csv.DictReader(csv_file_reader)
            index = 0
            for row in reader:
                index += 1
                if os.path.exists(row[csv_path_col_name]):
                    pool.apply_async(inference_function, args=(model, row, result_queue, index, test_transform, preprocessor, palette, output_dir, csv_path_col_name, device))
                    # inference_function(model, row, result_queue, index, test_transform, preprocessor, palette, output_dir, csv_path_col_name, device)
                # if index > 100:
                #     break

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

    all_rows = []
    while not result_queue.empty():
        row = result_queue.get()
        all_rows.append(row)

    # Write the filtered data to a new CSV file
    print(all_rows)
    with open(output_csv_file, 'w', newline='') as output_csv:
        fieldnames = all_rows[0].keys()
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Inference on all images completed.")


if __name__ == '__main__':
    main()