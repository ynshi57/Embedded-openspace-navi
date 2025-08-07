import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt


class FreespaceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform

        self.img_tf = T.Compose([
            T.Resize((240, 320)),
            T.ToTensor(),
        ])
        self.mask_tf = T.Compose([
            T.Resize((240, 320)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_tf(image)
        mask = self.mask_tf(mask)
        mask  = (mask > 0.5).float()

        return image, mask

    def plot_prediction(image, mask, pred):
        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()
        pred = pred.squeeze().detach().numpy()
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image)
        axs[0].set_title('Input Image')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(pred > 0.5, cmap='gray')
        axs[2].set_title('Prediction')
        plt.show()