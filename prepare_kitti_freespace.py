import os
import random
import tarfile
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"
ZIP_NAME = "kitti_road.zip"
EXTRACT_DIR = "kitti_road"
OUTPUT_DIR = "freespace_dataset"
NUM_SAMPLES = 100  # 你可以改为50或更多

def download_kitti(url, save_path):
    if os.path.exists(save_path):
        print(f"[✓] 文件已存在：{save_path}")
        return

    print("[↓] 正在下载 KITTI Road 数据集...,save :",save_path)
    urllib.request.urlretrieve(url, save_path)
    print("[✓] 下载完成。")

def extract_zip(zip_path, extract_to):
    import zipfile
    print("[✂️] 解压文件...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[✓] 解压完成。")

def add_road_after_all_underscores(input_string):
    """
    在字符串中找到所有的"_"字符，然后在每个后面加上"road_"
    
    Args:
        input_string (str): 输入的字符串
        
    Returns:
        str: 处理后的字符串
    """
    result = input_string
    offset = 0
    
    for i, char in enumerate(input_string):
        if char == "_":
            # 计算在当前结果字符串中的实际位置
            actual_index = i + offset
            # 在"_"后面插入"road_"
            result = result[:actual_index + 1] + "road_" + result[actual_index + 1:]
            # 更新偏移量，因为插入了5个字符
            offset += 5
    
    return result

def collect_samples(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, num_samples):
    all_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith(".png")])
    random.seed(42)
    sample_files = random.sample(all_files, num_samples)

    for idx, filename in enumerate(sample_files):
        img_src = os.path.join(src_img_dir, filename)
        mask_src = os.path.join(src_mask_dir, add_road_after_all_underscores(filename))

        img_dst = os.path.join(dst_img_dir, f"{idx:04d}.png")
        mask_dst = os.path.join(dst_mask_dir, f"{idx:04d}.png")

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(mask_src, mask_dst)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)

    zip_path = ZIP_NAME
    extract_path = EXTRACT_DIR

    # 下载
    print("⚙️ 第一步：下载数据集...")
    print("🚨 注意：该数据集是 zip 文件，不是 tar.gz，请确认 unzip 工具正常安装。")
    download_kitti(KITTI_URL, zip_path)

    # 解压
    # extract_zip(zip_path, extract_path)

    # 路径配置
    src_img_dir = os.path.join(extract_path, "data_road", "training", "image_2")
    src_mask_dir = os.path.join(extract_path, "data_road", "training", "gt_image_2")

    # 拷贝样本
    print(f"📦 第二步：采样并整理 {NUM_SAMPLES} 张图像...")
    collect_samples(src_img_dir, src_mask_dir, f"{OUTPUT_DIR}/images", f"{OUTPUT_DIR}/masks", NUM_SAMPLES)

    print("✅ 数据准备完成！保存路径：", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()
