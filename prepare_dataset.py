import argparse
import torchvision.transforms.functional as func
from PIL import Image
import os
import subprocess
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

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
      "/content/dataset/"
    ])

def create_high_resolution_samples_from_image(image_path, size, test=False):
    k = size // 2
    img = np.asarray(Image.open(image_path).convert('RGB'))
    for i in range((img.shape[0]-size)//k):
        for j in range((img.shape[1]-size)//k):
            new_img = Image.fromarray(
              img[i*k:i*k+size, j*k:j*k+size],
              mode="RGB"
            )
            if not test:
                new_img.save(
                  "/content/dataset/origin/"\
                  +image_path.split("/")[-1].split(".")[0]+f"_crop_{i}_{j}.png"
                )
            else:
                new_img.save(
                  "/content/dataset/test/"\
                  +image_path.split("/")[-1].split(".")[0]+f"_crop_{i}_{j}.png"
                )


def create_high_resolution_data(size):
    os.makedirs("/content/dataset/origin/")
    os.makedirs("/content/dataset/test/")
    files_path = [
      entry.path 
      for entry in os.scandir("/content/dataset/DIV2K_train_HR") 
      if entry.is_file()
    ]
    result = Parallel(n_jobs=4)(
      delayed(create_high_resolution_samples_from_image)(image_path, size) 
      for image_path in tqdm(
        files_path[:int(len(files_path)*0.9)], 
        desc="train data generation"
      )
    )
    result = Parallel(n_jobs=4)(
      delayed(create_high_resolution_samples_from_image)(image_path, size,True) 
      for image_path in tqdm(
        files_path[int(len(files_path)*0.9):], 
        desc="test data generation"
      )
    )

def resize_image(img_path, sizes, interpolation):
    img = Image.open(img_path)
    img = img.convert('RGB')
    lr_img = func.resize(
      img, 
      size = sizes[1], 
      interpolation  = interpolation
    )
    hr_lr_img = func.resize(
      lr_img, 
      size = sizes[0], 
      interpolation  = interpolation
    )
    lr_img.save(
      '/content/dataset/lr_data/{}'.format(img_path.split("/")[-1])
    )
    hr_lr_img.save(
      '/content/dataset/hr_lr_data/{}'.format(img_path.split("/")[-1])
    )

def prepare_data(sizes, interpolation):
    files_path = [
      entry.path 
      for entry in os.scandir("/content/dataset/origin") 
      if entry.is_file()
    ]
    os.makedirs("/content/dataset/lr_data/")
    os.makedirs("/content/dataset/hr_lr_data/")
    
    result = Parallel(n_jobs=4)(
      delayed(resize_image)(image_path, sizes, interpolation) 
      for image_path in tqdm(
        files_path, 
        desc="train data resized"
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

    create_high_resolution_data(args.high)

    prepare_data((args.high,args.low), resample_map[args.resample])