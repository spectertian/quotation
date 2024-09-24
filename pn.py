import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import svgwrite
import ezdxf
from tqdm import tqdm
import time
import math
import io
import tempfile
import os

def image_to_dots(image_path, min_diameter=0.1, density_factor=1.0):
    print(f"正在处理图像: {image_path}")
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    height, width = img_array.shape
    dot_size = max(1, int(min_diameter * 3.779528 * density_factor))
    dots = []

    total_steps = (height // dot_size) * (width // dot_size)
    with tqdm(total=total_steps, desc="转换图像为圆点") as pbar:
        for y in range(0, height, dot_size):
            for x in range(0, width, dot_size):
                if x + dot_size <= width and y + dot_size <= height:
                    avg_gray = np.mean(img_array[y:y+dot_size, x:x+dot_size])
                    dot_diameter = (255 - avg_gray) / 255 * dot_size
                    if dot_diameter > 0:
                        dots.append((x + dot_size/2, y + dot_size/2, dot_diameter))
                pbar.update(1)

    print(f"图像处理完成，共生成 {len(dots)} 个圆点")
    return dots, width, height


def save_eps_optimized(dots, width, height, filename):
    print(f"正在保存 EPS 文件: {filename}")

    # EPS 文件头
    eps_header = f"""%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 {width} {height}
%%Pages: 1
%%DocumentData: Clean7Bit
%%LanguageLevel: 2
%%EndComments
%%BeginProlog
/cp {{newpath 0 360 arc closepath fill}} bind def
%%EndProlog
%%Page: 1 1
"""

    # EPS 文件尾
    eps_footer = "%%EOF\n"

    total_dots = len(dots)
    batch_size = 10000
    num_batches = total_dots // batch_size + (1 if total_dots % batch_size != 0 else 0)

    with open(filename, 'w') as eps_file:
        eps_file.write(eps_header)

        with tqdm(total=total_dots + 100, desc="绘制圆点") as pbar:
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_dots)
                batch = dots[start:end]

                for x, y, d in batch:
                    eps_file.write(f"{x} {height - y} {d / 2} cp\n")
                    pbar.update(1)

                eps_file.flush()

            pbar.set_description("完成文件")
            for _ in range(100):
                time.sleep(0.01)
                pbar.update(1)

        eps_file.write(eps_footer)

    print(f"EPS 文件保存完成: {filename}")
def save_eps(dots, width, height, filename):
    print(f"正在保存 EPS 文件: {filename}")
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    with tqdm(total=len(dots) + 100, desc="绘制圆点") as pbar:
        for x, y, d in dots:
            circle = plt.Circle((x, height - y), d / 2, fill=True, color='black')
            plt.gca().add_artist(circle)
            pbar.update(1)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        pbar.set_description("准备保存")
        for _ in range(50):
            time.sleep(0.01)
            pbar.update(1)

        # 使用临时文件来保存 EPS
        with tempfile.NamedTemporaryFile(delete=False, suffix='.eps') as temp_file:
            temp_filename = temp_file.name

        plt.savefig(temp_filename, format='eps', bbox_inches='tight', pad_inches=0)

        pbar.set_description("写入文件")
        file_size = os.path.getsize(temp_filename)
        with open(temp_filename, 'rb') as temp_file, open(filename, 'wb') as output_file:
            for chunk in iter(lambda: temp_file.read(4096), b''):
                output_file.write(chunk)
                pbar.update(50 * len(chunk) / file_size)

        os.unlink(temp_filename)  # 删除临时文件

    plt.close()
    print(f"EPS 文件保存完成: {filename}")


def save_plt(dots, width, height, filename):
    print(f"正在保存 PLT 文件: {filename}")
    total_dots = len(dots)
    batch_size = 10000
    num_batches = total_dots // batch_size + (1 if total_dots % batch_size != 0 else 0)

    with open(filename, 'w') as plt_file:
        plt_file.write("IN;\nSP1;\n")
        with tqdm(total=total_dots + 100, desc="绘制圆点") as pbar:
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_dots)
                batch = dots[start:end]
                for x, y, d in batch:
                    plt_x, plt_y = int(x * 40), int((height - y) * 40)
                    radius = int(d * 20)
                    plt_file.write(f"PU{plt_x},{plt_y};\n")
                    for j in range(37):
                        angle = j * 10 * math.pi / 180
                        circle_x = int(plt_x + radius * math.cos(angle))
                        circle_y = int(plt_y + radius * math.sin(angle))
                        plt_file.write(f"{'PD' if j == 0 else ''}{circle_x},{circle_y};\n")
                    plt_file.write("PU;\n")
                    pbar.update(1)
                plt_file.flush()
            plt_file.write("SP0;\nIN;\n")
            pbar.set_description("完成文件")
            for _ in range(100):
                time.sleep(0.01)
                pbar.update(1)
    print(f"PLT 文件保存完成: {filename}")


def save_svg(dots, width, height, filename):
    print(f"正在保存 SVG 文件: {filename}")
    dwg = svgwrite.Drawing(filename, size=(width, height))
    total_dots = len(dots)
    batch_size = 10000
    num_batches = total_dots // batch_size + (1 if total_dots % batch_size != 0 else 0)

    with tqdm(total=total_dots + 100, desc="绘制圆点") as pbar:
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_dots)
            batch = dots[start:end]
            for x, y, d in batch:
                dwg.add(dwg.circle(center=(x, y), r=d / 2, fill='black'))
                pbar.update(1)
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                dwg.save()
        pbar.set_description("完成文件")
        for _ in range(100):
            time.sleep(0.01)
            pbar.update(1)
    print(f"SVG 文件保存完成: {filename}")


def save_dxf(dots, filename):
    print(f"正在保存 DXF 文件: {filename}")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    total_dots = len(dots)
    with tqdm(total=total_dots + 100, desc="绘制圆点") as pbar:
        for x, y, d in dots:
            msp.add_circle((x, y), d / 2)
            pbar.update(1)
        pbar.set_description("写入文件")
        for _ in range(100):
            time.sleep(0.01)
            pbar.update(1)
    doc.saveas(filename)
    print(f"DXF 文件保存完成: {filename}")


def save_ai(dots, width, height, filename):
    print(f"正在保存 AI 文件 (PDF 格式): {filename}")
    plt.figure(figsize=(width / 100, height / 100), dpi=300)
    with tqdm(total=len(dots) + 100, desc="绘制圆点") as pbar:
        for x, y, d in dots:
            circle = plt.Circle((x, height - y), d / 2, fill=True, color='black')
            plt.gca().add_artist(circle)
            pbar.update(1)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        pbar.set_description("准备保存")
        for _ in range(50):
            time.sleep(0.01)
            pbar.update(1)

        # 使用临时文件来保存 PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_filename = temp_file.name

        plt.savefig(temp_filename, format='pdf', bbox_inches='tight', pad_inches=0)

        pbar.set_description("写入文件")
        file_size = os.path.getsize(temp_filename)
        with open(temp_filename, 'rb') as temp_file, open(filename, 'wb') as output_file:
            for chunk in iter(lambda: temp_file.read(4096), b''):
                output_file.write(chunk)
                pbar.update(50 * len(chunk) / file_size)

        os.unlink(temp_filename)  # 删除临时文件

    plt.close()
    print(f"AI 文件 (PDF 格式) 保存完成: {filename}")


# 在主函数中调用这些函数
def main():
    image_path = '1.png'  # 请替换为您的输入图像路径
    min_diameter = 0.1  # 最小圆直径（毫米）
    density_factor = 1.0  # 密度因子，增加会使圆点更稀疏

    dots, width, height = image_to_dots(image_path, min_diameter, density_factor)

    # save_eps(dots, width, height, 'output.eps')
    # save_eps_optimized(dots, width, height, 'output_optimized.eps')

    # save_plt(dots, width, height, 'output.plt')
    # save_svg(dots, width, height, 'output.svg')
    # save_dxf(dots, 'output.dxf')
    save_ai(dots, width, height, 'output.pdf')  # AI 格式实际上保存为 PDF


if __name__ == "__main__":
    main()