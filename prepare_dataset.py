import argparse
import torchvision.transforms.functional as func
from PIL import Image
import os
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm

PATH_PREFIX = os.path.join(os.path.dirname(__file__), "dataset/")

def download_data():
    subprocess.run([
      "wget", 
      "-q", 
      "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    ])
    subprocess.run([
      "unzip", 
      "-q", 
      "DIV2K_train_HR.zip", 
      "-d", 
      os.path.join(os.path.dirname(__file__), "dataset/")
    ])

def create_high_resolution_samples_from_image(image_path, size, interpolation, test=False):
    img = Image.open(image_path).convert('RGB')
    new_img = func.resize(
      img, 
      size = (size, size), 
      interpolation  = interpolation
    )
    if not test:
        new_img.save(
          PATH_PREFIX+"train/"\
          +image_path.split("/")[-1]
        )
    else:
        new_img.save(
          PATH_PREFIX+"test/"\
          +image_path.split("/")[-1]
        )


def create_high_resolution_data(size, interpolation):
    os.makedirs(PATH_PREFIX+"train/")
    os.makedirs(PATH_PREFIX+"test/")
    files_path = [
      entry.path 
      for entry in os.scandir(PATH_PREFIX+"DIV2K_train_HR") 
      if entry.is_file()
    ]
    result = Parallel(n_jobs=4)(
      delayed(create_high_resolution_samples_from_image)(image_path, size, interpolation) 
      for image_path in tqdm(
        files_path[:int(len(files_path)*0.9)], 
        desc="train data generation"
      )
    )
    result = Parallel(n_jobs=4)(
      delayed(create_high_resolution_samples_from_image)(image_path, size, interpolation,True) 
      for image_path in tqdm(
        files_path[int(len(files_path)*0.9):], 
        desc="test data generation"
      )
    )

def resize_image(img_path, sizes, interpolation, test=False):
    prefix = "test_" if test else "train_"
    img = Image.open(img_path)
    img = img.convert('RGB')
    lr_img = func.resize(
      img, 
      size = (sizes[1], sizes[1]), 
      interpolation  = interpolation
    )
    hr_lr_img = func.resize(
      lr_img, 
      size = (sizes[0], sizes[0]), 
      interpolation  = interpolation
    )
    lr_img.save(
      PATH_PREFIX+f'{prefix}lr_data/{img_path.split("/")[-1]}'
    )
    hr_lr_img.save(
      PATH_PREFIX+f'{prefix}hr_lr_data/{img_path.split("/")[-1]}'
    )

def prepare_data(sizes, interpolation):
    files_path = [
      entry.path 
      for entry in os.scandir(PATH_PREFIX+"train") 
      if entry.is_file()
    ]
    os.makedirs(PATH_PREFIX+"train_lr_data/")
    os.makedirs(PATH_PREFIX+"train_hr_lr_data/")
    
    result = Parallel(n_jobs=4)(
      delayed(resize_image)(image_path, sizes, interpolation) 
      for image_path in tqdm(
        files_path, 
        desc="train data resized"
      )
    )


    files_path = [
      entry.path 
      for entry in os.scandir(PATH_PREFIX+"test") 
      if entry.is_file()
    ]
    os.makedirs(PATH_PREFIX+"test_lr_data/")
    os.makedirs(PATH_PREFIX+"test_hr_lr_data/")

    result = Parallel(n_jobs=4)(
      delayed(resize_image)(image_path, sizes, interpolation, test=True) 
      for image_path in tqdm(
        files_path, 
        desc="test data resized"
      )
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--high", "-H", type=int)
    parser.add_argument("--low", "-l", type=int)
    parser.add_argument("--resample",type=str, default="bicubic")
    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}

    args = parser.parse_args()

    download_data()

    create_high_resolution_data(args.high, resample_map[args.resample])

    prepare_data((args.high,args.low), resample_map[args.resample])