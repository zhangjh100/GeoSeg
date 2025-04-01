import cv2
import numpy as np
from pathlib import Path


def convert_label_mask(mask, label_mapping):
    """
    根据提供的标签映射字典，将标签掩码中的像素值替换为对应的类别值。

    参数:
        mask (np.ndarray): 输入的标签掩码图像（numpy数组）。
        label_mapping (dict): 从原始像素值到新类别值的映射字典。

    返回:
        np.ndarray: 转换后的标签掩码图像。
    """
    # 创建一个新的掩码副本以避免修改原始数据
    converted_mask = mask.copy()

    # 遍历映射字典并替换像素值
    for old_value, new_value in label_mapping.items():
        converted_mask[mask == old_value] = new_value

    return converted_mask


def process_masks(input_dir, output_dir, label_mapping):
    """
    处理输入目录下的所有标签掩码图像，并将它们保存到输出目录中。

    参数:
        input_dir (str): 包含标签掩码图像的输入目录路径。
        output_dir (str): 输出目录路径。
        label_mapping (dict): 从原始像素值到新类别值的映射字典。
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取输入目录下所有的png或tif文件
    mask_paths = list(Path(input_dir).glob('*.png')) + list(Path(input_dir).glob('*.tif'))

    for mask_path in mask_paths:
        # 读取掩码图像
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"无法读取文件：{mask_path}")
            continue

        # 转换标签掩码
        converted_mask = convert_label_mask(mask, label_mapping)

        # 构造输出路径
        output_path = Path(output_dir) / mask_path.name

        # 保存转换后的掩码图像
        cv2.imwrite(str(output_path), converted_mask)
        print(f"已处理并保存: {output_path}")


# 示例用法
if __name__ == "__main__":
    input_dir = './data/acdc/train_masks'  # 替换为你的输入目录路径
    output_dir = './data/acdc/train/train_masks_convert'  # 替换为你的输出目录路径
    label_mapping = {
        0: 0,  # 将像素值为4的区域改为类别0
        171: 1,  # 将像素值为170的区域改为类别1
        114: 2,  # 将像素值为113的区域改为类别2
        57: 3  # 将像素值为56的区域改为类别3
    }

    process_masks(input_dir, output_dir, label_mapping)