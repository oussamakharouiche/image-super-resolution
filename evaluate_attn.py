import os
import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as func
from PIL import Image
from model.sr3 import Gaussiendiffusion, linear_beta_schedule
from load_dataset import PairedImageDatasetTest
from model.components import Unet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math


CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoint_new/")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_new/")
MODEL_FILENAME = "model_with_attn_params_1000.pth"
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, MODEL_FILENAME)

T = 1000
UNET_IN_CHANNELS = 6
UNET_OUT_CHANNELS = 3
UNET_WITH_ATTN = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_hr_folder = DATASET_DIR+"test_hr_lr_data"
test_lr_folder = DATASET_DIR+"test_hr_lr_data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = PairedImageDatasetTest(test_hr_folder, test_lr_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

unet = Unet(T, UNET_IN_CHANNELS, UNET_OUT_CHANNELS, with_attn=UNET_WITH_ATTN).to(device)
unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))


def print_norm(name, image):
    min_val = image.min().item()
    max_val = image.max().item()

    l2_norm = torch.norm(image).item()

    print(f"{name} : min value {min_val} | max value {max_val} | norm {l2_norm}")


def run_inference(model, dataloader, device, T=T, output_dir="test_output_attn"):
    os.makedirs(DATASET_DIR+output_dir, exist_ok=True)
    betas, cumulative_alphas = linear_beta_schedule(T)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (hr_imgs, lr_imgs, paths) in enumerate(tqdm(test_loader)):
            lr_imgs = lr_imgs.to(device)  # [B, 3, H, W]

            
            print_norm("lr_imgs", lr_imgs)

            batch_size = lr_imgs.shape[0]
            
            img_pred = torch.randn((batch_size, 3, 256, 256), device=device)
            
            for t in list(reversed(range(T))):
                x = torch.cat([lr_imgs, img_pred], dim=1)  # [B, 6, H, W]
                
                # Predict noise
                noise_p = model(x, torch.tensor([t]).repeat(batch_size).to(device))

                print_norm("noise", noise_p)
                
                # Update predictions
                img_pred = (1 / math.sqrt(1 - betas[t])) * (
                    img_pred - (betas[t] / math.sqrt(1 - cumulative_alphas[t])) * noise_p
                )

                if t > 0:
                    img_pred += math.sqrt(betas[t]) * torch.randn_like(img_pred)

                img_pred = torch.clamp(img_pred, -1.0, 1.0)
                print_norm("pred",img_pred)
            
            img_pred = img_pred * 0.5 + 0.5
            img_pred = torch.clamp(img_pred, 0.0, 1.0)
            
            for i in range(batch_size):
                img_pil = transforms.ToPILImage()(img_pred[i].cpu())
                output_path = os.path.join(DATASET_DIR+output_dir, paths[i].split("/")[-1])
                img_pil.save(output_path) 



run_inference(unet, test_loader, device, T)