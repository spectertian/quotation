import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import ezdxf
import os
import random
import time
from tqdm import tqdm


def read_bmp(file_path):
    print(f"Reading image: {file_path}")
    with Image.open(file_path) as img:
        width, height = img.size
        gray_img = img.convert('L')
        print(f"Image size: {width}x{height}")
        return np.array(gray_img), width, height


def create_dot_image(gray_img, min_dot_size_pixels, max_dot_size_pixels, min_spacing, max_spacing):
    height, width = gray_img.shape
    dot_image = np.zeros((height, width))
    total_pixels = height * width
    processed_pixels = 0
    start_time = time.time()

    print("Creating dot image...")
    with tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', ncols=70) as pbar:
        for y in range(0, height, min_spacing):
            for x in range(0, width, min_spacing):
                if y < height and x < width:
                    intensity = gray_img[y, x] / 255.0
                    if intensity < 1:  # 不是纯白色
                        dot_probability = 1 - intensity
                        if random.random() < dot_probability:
                            # 使用 min_dot_size_pixels 作为最小点大小
                            dot_size = int(
                                min_dot_size_pixels + (max_dot_size_pixels - min_dot_size_pixels) * (1 - intensity))
                            spacing = int(min_spacing + (max_spacing - min_spacing) * intensity)

                            if x % spacing == 0 and y % spacing == 0:
                                for dy in range(-dot_size // 2, dot_size // 2 + 1):
                                    for dx in range(-dot_size // 2, dot_size // 2 + 1):
                                        if dx * dx + dy * dy <= (dot_size // 2) ** 2:
                                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                                dot_image[y + dy, x + dx] = 1

                processed_pixels += min_spacing * min_spacing
                if processed_pixels >= total_pixels / 100:
                    pbar.update(1)
                    processed_pixels = 0

    end_time = time.time()
    print(f"Dot image creation completed in {end_time - start_time:.2f} seconds")
    return dot_image

def save_image(dot_image, filename, dpi):
    print(f"Saving image as {filename}...")
    fig = Figure(figsize=(dot_image.shape[1] / dpi, dot_image.shape[0] / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(dot_image, cmap='gray', interpolation='nearest')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存不同格式的文件
    print("Saving EPS file...")
    canvas.print_figure(f"{filename}.eps", dpi=dpi, format='eps')
    print("Saving PNG file...")
    canvas.print_figure(f"{filename}.png", dpi=dpi, format='png')
    print("Saving BMP file...")
    plt.imsave(f"{filename}.bmp", dot_image, cmap='gray', dpi=dpi)
    print("Saving PDF file...")
    fig.savefig(f"{filename}.pdf", dpi=dpi, format='pdf')

    # 保存AI文件 (实际上是PDF格式，可以用Adobe Illustrator打开)
    print("Saving AI file...")
    fig.savefig(f"{filename}.ai", dpi=dpi, format='pdf')

    print("All image files saved successfully")


def save_dxf(dot_image, filename, pixel_size):
    print(f"Saving DXF file as {filename}.dxf...")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    total_dots = np.sum(dot_image)
    dots_processed = 0

    with tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', ncols=70) as pbar:
        for y in range(dot_image.shape[0]):
            for x in range(dot_image.shape[1]):
                if dot_image[y, x] == 1:
                    center = (x * pixel_size, (dot_image.shape[0] - y - 1) * pixel_size)
                    msp.add_circle(center, radius=pixel_size / 2)
                    dots_processed += 1
                    if dots_processed >= total_dots / 100:
                        pbar.update(1)
                        dots_processed = 0

    doc.saveas(f"{filename}.dxf")
    print("DXF file saved successfully")


def main():
    input_file = "3.bmp"  # 替换为您的输入文件路径
    output_file = "output"

    gray_img, width, height = read_bmp(input_file)

    # 设置参数
    dpi = 300  # 输出DPI
    min_dot_size = 0.1  # 最小圆点直径（毫米）
    max_dot_size = 0.5  # 最大圆点直径（毫米）
    min_dot_size_pixels = int(min_dot_size / 25.4 * dpi)  # 将毫米转换为像素
    max_dot_size_pixels = int(max_dot_size / 25.4 * dpi)  # 将毫米转换为像素
    min_spacing = max_dot_size_pixels  # 最小间距等于最大点大小
    max_spacing = int(min_spacing * 3)  # 最大间距是最小间距的3倍

    print(
        f"Parameters: min_dot_size={min_dot_size}mm ({min_dot_size_pixels}px), max_dot_size={max_dot_size}mm ({max_dot_size_pixels}px)")
    print(f"min_spacing={min_spacing}px, max_spacing={max_spacing}px")

    dot_image = create_dot_image(gray_img, min_dot_size_pixels, max_dot_size_pixels, min_spacing, max_spacing)

    save_image(dot_image, output_file, dpi)
    save_dxf(dot_image, output_file, 1 / dpi * 25.4)  # 将像素大小转换为毫米

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()