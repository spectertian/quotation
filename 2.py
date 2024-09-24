import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import svgwrite
import ezdxf
import math


def image_to_dots(image_path, min_diameter=0.1, density_factor=1.0, scale_factor=0.5):
    img = Image.open(image_path).convert('L')

    # 缩放图像
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    img_array = np.array(img)

    height, width = img_array.shape
    dot_size = max(1, int(min_diameter * 3.779528 * density_factor))
    dots = []

    for y in range(0, height, dot_size):
        for x in range(0, width, dot_size):
            if x + dot_size <= width and y + dot_size <= height:
                avg_gray = np.mean(img_array[y:y + dot_size, x:x + dot_size])
                dot_diameter = (255 - avg_gray) / 255 * dot_size
                if dot_diameter > 0:
                    dots.append((x + dot_size / 2, y + dot_size / 2, dot_diameter))

    return dots, width, height


def save_eps_optimized(dots, width, height, filename):
    print(f"正在保存 EPS 文件: {filename}")

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
    eps_footer = "%%EOF\n"

    with open(filename, 'w') as eps_file:
        eps_file.write(eps_header)

        with tqdm(total=len(dots), desc="绘制圆点") as pbar:
            for x, y, d in dots:
                eps_file.write(f"{x} {height - y} {d / 2} cp\n")
                pbar.update(1)

        eps_file.write(eps_footer)

    print(f"EPS 文件保存完成: {filename}")


def save_plt(dots, width, height, filename):
    print(f"正在保存 PLT 文件: {filename}")
    with open(filename, 'w') as plt_file:
        plt_file.write("IN;\nSP1;\n")
        with tqdm(total=len(dots), desc="绘制圆点") as pbar:
            for x, y, d in dots:
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
        plt_file.write("SP0;\nIN;\n")
    print(f"PLT 文件保存完成: {filename}")


def save_svg(dots, width, height, filename):
    print(f"正在保存 SVG 文件: {filename}")
    dwg = svgwrite.Drawing(filename, size=(width, height))
    with tqdm(total=len(dots), desc="绘制圆点") as pbar:
        for x, y, d in dots:
            dwg.add(dwg.circle(center=(x, y), r=d / 2, fill='black'))
            pbar.update(1)
    dwg.save()
    print(f"SVG 文件保存完成: {filename}")


def save_dxf(dots, filename):
    print(f"正在保存 DXF 文件: {filename}")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    with tqdm(total=len(dots), desc="绘制圆点") as pbar:
        for x, y, d in dots:
            msp.add_circle((x, y), d / 2)
            pbar.update(1)
    doc.saveas(filename)
    print(f"DXF 文件保存完成: {filename}")


# 主函数
def main():
    image_path = '1.png'  # 请替换为您的输入图像路径
    min_diameter = 0.1  # 最小圆直径（毫米）
    density_factor = 1.0  # 密度因子，增加会使圆点更稀疏
    scale_factor = 0.5  # 缩放因子，减小这个值会使输出文件更小

    dots, width, height = image_to_dots(image_path, min_diameter, density_factor, scale_factor)

    save_eps_optimized(dots, width, height, 'output_small.eps')
    save_plt(dots, width, height, 'output_small.plt')
    save_svg(dots, width, height, 'output_small.svg')
    save_dxf(dots, 'output_small.dxf')


if __name__ == "__main__":
    main()