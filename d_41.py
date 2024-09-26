from PIL import Image, ImageDraw
import random
import math


def mm_to_pixels(mm, dpi=300):
    return int(mm * dpi / 25.4)


def create_density_gradient_dot_image(input_path, output_path, min_dot_size_mm=0.1, max_dot_size_mm=1,
                                      density_factor=1, dpi=300, white_threshold=245):
    # 打开原始图片
    original_image = Image.open(input_path)

    # 转换为灰度图
    grayscale_image = original_image.convert('L')

    # 创建一个新的白色背景图片
    dot_image = Image.new('RGB', original_image.size, (255, 255, 255))
    draw = ImageDraw.Draw(dot_image)

    # 转换毫米到像素
    min_dot_size = mm_to_pixels(min_dot_size_mm, dpi)
    max_dot_size = mm_to_pixels(max_dot_size_mm, dpi)

    # 定义区块大小
    block_size = max(max_dot_size * 2, 10)

    total_blocks = ((grayscale_image.width + block_size - 1) // block_size) * \
                   ((grayscale_image.height + block_size - 1) // block_size)
    processed_blocks = 0

    # 遍历图片的每个区块
    for x in range(0, grayscale_image.width, block_size):
        for y in range(0, grayscale_image.height, block_size):
            # 获取当前区块的平均灰度值
            block = grayscale_image.crop((x, y, min(x + block_size, grayscale_image.width),
                                          min(y + block_size, grayscale_image.height)))
            avg_gray = sum(block.getdata()) / len(block.getdata())

            # 如果平均灰度值高于白色阈值，跳过这个区块
            if avg_gray > white_threshold:
                processed_blocks += 1
                continue

            # 计算这个区块的圆点大小
            dot_size = max_dot_size - (avg_gray / 255.0) * (max_dot_size - min_dot_size)
            dot_size = max(min_dot_size, min(max_dot_size, dot_size))

            # 计算这个区块应该有多少个圆点
            area_ratio = 1 - (avg_gray / 255.0)
            max_dots = int((block_size ** 2) / (dot_size ** 2) * area_ratio * density_factor)
            dot_count = random.randint(max(1, max_dots // 2), max_dots)

            # 在这个区块内随机放置圆点
            for _ in range(dot_count):
                dot_x = x + random.randint(0, block_size - math.ceil(dot_size))
                dot_y = y + random.randint(0, block_size - math.ceil(dot_size))
                draw.ellipse([dot_x, dot_y, dot_x + dot_size, dot_y + dot_size], fill=(0, 0, 0))

            processed_blocks += 1
            if processed_blocks % 100 == 0 or processed_blocks == total_blocks:
                print(
                    f"Progress: {processed_blocks}/{total_blocks} blocks processed ({processed_blocks / total_blocks * 100:.2f}%)")

    # 保存结果图片
    dot_image.save(output_path)
    print("Image processing completed.")


# 使用示例
input_image = "3.bmp"
output_image = "output_density_gradient.png"
create_density_gradient_dot_image(input_image, output_image,
                                  min_dot_size_mm=0.1, max_dot_size_mm=1,
                                  density_factor=1, dpi=300, white_threshold=245)