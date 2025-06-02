import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from load_dataset import PairedImageDataset
from model.sr3 import Gaussiendiffusion
from model.components import Unet


PATH_PREFIX = os.path.join(os.path.dirname(__file__), "dataset/")
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), "checkpoint/")
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hr_folder = PATH_PREFIX+"train"
    lr_folder = PATH_PREFIX+"train_hr_lr_data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = PairedImageDataset(hr_folder, lr_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    T = 1000

    unet = Unet(1000, 6, 3, with_attn=False).to(device)
    model = Gaussiendiffusion(unet, T, device, schedule="linear")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    warmup_steps = 10000
    def linear_warmup(step):
        return min(1.0, step / warmup_steps)

    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup)

    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20, 
        verbose=True
    )

    # Training loop
    num_epochs = 1000
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for batch_idx, (x_hr, x_lr) in enumerate(dataloader):
            x_hr, x_lr = x_hr.to(device), x_lr.to(device)

            # Compute diffusion loss
            loss = model(x_hr, x_lr)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step = epoch * len(dataloader) + batch_idx
            if current_step < warmup_steps:
                scheduler_warmup.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss / len(dataloader):.4f}")



        if (epoch + 1) % 50 == 0:
          model_path = os.path.join(CHECKPOINTS_PATH, f"model_params_{epoch+1}.pth")
          torch.save(unet.state_dict(), model_path)
          print(f"Model saved")
