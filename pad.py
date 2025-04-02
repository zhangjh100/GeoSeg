import os
from PIL import Image
import glob


def resize_images(input_dir, output_dir, target_size=(512, 512)):
    """
    将输入目录下的所有图像从256x256扩展到512x512，并保存到输出目录。

    参数:
        input_dir (str): 包含原始图像的输入目录路径。
        output_dir (str): 输出目录路径。
        target_size (tuple): 目标图像尺寸，默认为(512, 512)。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式列表
    supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

    # 遍历支持的所有格式
    for fmt in supported_formats:
        # 获取输入目录下所有匹配的文件路径
        image_paths = glob.glob(os.path.join(input_dir, fmt))

        for img_path in image_paths:
            # 打开图像
            with Image.open(img_path) as img:
                # 检查图像尺寸是否为256x256
                if img.size == (256, 256):
                    # 创建一个新的空白图像，大小为目标尺寸，背景色为黑色
                    new_img = Image.new("RGB", target_size, "black")

                    # 计算原图放置在新图中的位置
                    paste_position = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)

                    # 将原图粘贴到新图中央
                    new_img.paste(img, paste_position)

                    # 构造输出路径
                    output_path = os.path.join(output_dir, os.path.basename(img_path))

                    # 保存新图像
                    new_img.save(output_path)
                    print(f"已处理并保存: {output_path}")
                else:
                    print(f"跳过非256x256图像: {img_path}")


# 示例用法
if __name__ == "__main__":
    input_dir = 'data/acdc/train/train_masks_convert'  # 替换为你的输入目录路径
    output_dir = 'data/acdc/train/train_masks_convert_0'  # 替换为你的输出目录路径

    resize_images(input_dir, output_dir)