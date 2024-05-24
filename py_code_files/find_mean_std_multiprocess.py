import argparse
import numpy as np
from PIL import Image
import pandas as pd
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation of images.")
    parser.add_argument("--csv", required=True, help="CSV file containing file paths")
    parser.add_argument("--csv_path_col_name", required=True, help="Column name for the paths in the csv, ex: file_path or img_path")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the images (assume square)")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use (default: all)")
    return parser.parse_args()

def calculate_mean_std_parallel(img_path):
    img_array = np.array(Image.open(img_path)) / 255
    sum_channels = np.sum(img_array, axis=(0, 1))
    sum_squared_channels = np.sum(img_array ** 2, axis=(0, 1))

    return sum_channels, sum_squared_channels

def calculate_mean_std(image_paths, image_size=(224,224), num_processes=None):
    num_images = len(image_paths)
    sum_channels = np.zeros((3,), dtype=np.float64)
    sum_squared_channels = np.zeros((3,), dtype=np.float64)

    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_mean_std_parallel, image_paths)

    for sum_ch, sum_squared_ch in results:
        sum_channels += sum_ch
        sum_squared_channels += sum_squared_ch

    mean_channels = sum_channels / (num_images * image_size[1] * image_size[0])
    std_channels = np.sqrt(
        (sum_squared_channels / (num_images * image_size[1] * image_size[0])) - mean_channels ** 2
    )

    return mean_channels, std_channels

def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    image_paths = df[args.csv_path_col_name].tolist()

    print("Number of images:", len(image_paths))

    mean, std = calculate_mean_std(image_paths, image_size=(args.img_size, args.img_size), num_processes=args.num_processes)
    print("Mean:", mean)
    print("Standard Deviation:", std)


if __name__ == "__main__":
    main()
