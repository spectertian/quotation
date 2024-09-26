import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import ezdxf
import os


def read_bmp(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
        gray_img = img.convert('L')
        return np.array(gray_img), width, height


def create_dot_image(gray_img, dot_size, spacing):
    height, width = gray_img.shape
    dot_image = np.zeros((height, width))

    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            if y < height and x < width:
                intensity = 1 - gray_img[y, x] / 255.0
                dot_radius = int(dot_size * intensity / 2)
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        if dx * dx + dy * dy <= dot_radius * dot_radius:
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                dot_image[y + dy, x + dx] = 1

        # 显示进度
        print(f"Processing: {y / height * 100:.2f}%")

    return dot_image


def save_image(dot_image, filename, dpi):
    fig = Figure(figsize=(dot_image.shape[1] / dpi, dot_image.shape[0] / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(dot_image, cmap='gray', interpolation='nearest')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存不同格式的文件
    canvas.print_figure(f"{filename}.eps", dpi=dpi, format='eps')
    canvas.print_figure(f"{filename}.png", dpi=dpi, format='png')
    plt.imsave(f"{filename}.bmp", dot_image, cmap='gray', dpi=dpi)
    fig.savefig(f"{filename}.pdf", dpi=dpi, format='pdf')

    # 保存AI文件 (实际上是PDF格式，可以用Adobe Illustrator打开)
    fig.savefig(f"{filename}.ai", dpi=dpi, format='pdf')


def save_dxf(dot_image, filename, pixel_size):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for y in range(dot_image.shape[0]):
        for x in range(dot_image.shape[1]):
            if dot_image[y, x] == 1:
                center = (x * pixel_size, (dot_image.shape[0] - y - 1) * pixel_size)
                msp.add_circle(center, radius=pixel_size / 2)

    doc.saveas(f"{filename}.dxf")


def main():
    input_file = "3.bmp"  # 替换为您的输入文件路径
    output_file = "output"

    gray_img, width, height = read_bmp(input_file)

    # 设置参数
    dpi = 300  # 输出DPI
    min_dot_size = 0.1  # 最小圆点直径（毫米）
    dot_size_pixels = int(min_dot_size / 25.4 * dpi)  # 将毫米转换为像素
    spacing = 5  # 点之间的间距（像素）

    dot_image = create_dot_image(gray_img, dot_size_pixels, spacing)

    save_image(dot_image, output_file, dpi)
    save_dxf(dot_image, output_file, 1 / dpi * 25.4)  # 将像素大小转换为毫米

    print("All files have been saved successfully.")


if __name__ == "__main__":
    main()