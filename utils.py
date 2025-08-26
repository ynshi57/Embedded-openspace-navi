import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

def get_all_mask_colors(mask_dir):
    """
    统计mask目录下所有出现过的RGB颜色
    """
    color_set = set()
    for fname in os.listdir(mask_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            mask = Image.open(os.path.join(mask_dir, fname)).convert('RGB')
            mask_np = np.array(mask)
            colors = mask_np.reshape(-1, 3)
            for color in np.unique(colors, axis=0):
                color_set.add(tuple(color))
    return color_set

# def mask_rgb_to_class(mask, color2class):
#     """
#     将RGB mask图像（PIL或np.array）转换为类别索引mask
#     color2class: dict, 例如 {(255,0,255):0, (255,0,0):1, (0,0,0):2}
#     返回: torch.LongTensor, shape=[H,W]
#     """
#     if isinstance(mask, Image.Image):
#         mask_np = np.array(mask)
#     else:
#         mask_np = mask
#     class_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
#     for color, idx in color2class.items():
#         match = np.all(mask_np == color, axis=-1)
#         class_mask[match] = idx
#     return torch.from_numpy(class_mask).long()

class KittiRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=True, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.augment = augment
        self.init_shape_h = 224
        self.init_shape_w = 224

        self.img_tf = T.Compose([
            T.Resize((self.init_shape_h, self.init_shape_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.mask_tf = T.Resize(
            (self.init_shape_h, self.init_shape_w),
            interpolation=Image.NEAREST
        )

        # image-only augmentation
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Geometric augmentation applied consistently
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Color jitter on image only
        if self.augment:
            image = self.color_jitter(image)

        image = self.img_tf(image)
        mask = self.mask_tf(mask)
        
        # 检查mask的统计信息（转换为NumPy后再统计）
        if idx < 5:
            mask_np_dbg = np.array(mask)
            min_val = mask_np_dbg.min()
            max_val = mask_np_dbg.max()
            mean_val = mask_np_dbg.mean()
            unique_colors = np.unique(mask_np_dbg.reshape(-1, 3), axis=0)
            print(f"Image {idx}: mask min {min_val}, max {max_val}, mean {mean_val:.3f}, unique colors {unique_colors.tolist()}")

        mask = np.array(mask)

        label = np.zeros(mask.shape[:2], dtype=np.uint8)

        # 红色 -> 背景类 0
        label[(mask[:,:,0] == 255) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0)] = 0

        # 粉色 -> 道路类 1
        label[(mask[:,:,0] == 255) & (mask[:,:,1] == 0) & (mask[:,:,2] == 255)] = 1

        # 蓝色 -> 统一设为 背景类 0 （避免未映射颜色干扰）
        label[(mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 255)] = 0

        # 黑色 -> ignore 255
        label[(mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0)] = 255

        label = torch.from_numpy(label).long()  # (H,W)

        return image, label

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