from PIL import Image
from PIL import Image, ImageDraw
import svgwrite
import ezdxf
from tqdm import tqdm
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import gc
import random

def image_to_dots(image_path, min_diameter=0.1, density_factor=1.0, threshold=200):
    print("正在读取和处理图像...")
    img = Image.open(image_path).convert('RGB')  # 改为RGB模式

    img_array = np.array(img)
    height, width, _ = img_array.shape

    base_dot_size = max(1, int(min_diameter * 3.779528 * density_factor))

    dots = []
    total_steps = height * width

    with tqdm(total=total_steps, desc="转换图像为圆点") as pbar:
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x]
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                if brightness * 255 >= threshold:
                    continue  # 跳过很亮的区域

                max_color = max(r, g, b)
                min_color = min(r, g, b)
                saturation = (max_color - min_color) / max_color if max_color != 0 else 0

                density_adjustment = 1 + saturation
                probability = (1 - brightness) * density_adjustment

                if random.random() < probability * density_factor:
                    offset_x = random.uniform(-base_dot_size / 2, base_dot_size / 2)
                    offset_y = random.uniform(-base_dot_size / 2, base_dot_size / 2)
                    dot_x = x + offset_x
                    dot_y = y + offset_y
                    dot_diameter = base_dot_size * (1 - brightness) * random.uniform(0.8, 1.2)
                    dots.append((dot_x, dot_y, dot_diameter))

                pbar.update(1)

    print(f"处理完成，共生成 {len(dots)} 个圆点")
    return dots, width, height

def save_eps(dots, width, height, filename):
    print(f"正在保存 EPS 文件: {filename}")
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

    batch_size = 100000  # 每批处理的圆点数
    num_batches = len(dots) // batch_size + (1 if len(dots) % batch_size != 0 else 0)

    try:
        for i in tqdm(range(num_batches), desc="绘制 EPS"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(dots))
            batch = dots[start:end]
            x, y, d = zip(*batch)
            e = EllipseCollection(
                widths=d, heights=d, angles=0,
                units='xy', offsets=list(zip(x, [height - yi for yi in y])),
                transOffset=ax.transData,
                facecolors='black'
            )
            ax.add_collection(e)
            if i % 5 == 0:  # 每5批后清理内存
                gc.collect()

        print("正在保存 EPS 文件...")
        plt.savefig(filename, format='eps', bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"EPS 文件已成功保存: {filename}")
    except Exception as e:
        print(f"保存 EPS 文件时发生错误: {e}")
    finally:
        plt.close()
        gc.collect()

def save_plt(dots, width, height, filename):
    print(f"正在准备保存 PLT 文件: {filename}")
    total_dots = len(dots)
    batch_size = 10000  # 每批处理的圆点数
    num_batches = total_dots // batch_size + (1 if total_dots % batch_size != 0 else 0)

    with open(filename, 'w') as plt_file:
        plt_file.write("IN;\n")  # 初始化
        plt_file.write("SP1;\n")  # 选择笔 1
        with tqdm(total=total_dots, desc="整体进度") as pbar:
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_dots)
                batch = dots[start:end]
                batch_desc = f"批次 {i + 1}/{num_batches}"
                for x, y, d in tqdm(batch, desc=batch_desc, leave=False):
                    plt_x = int(x * 40)
                    plt_y = int((height - y) * 40)  # 翻转 Y 坐标
                    radius = int(d * 20)  # 直径的一半
                    plt_file.write(f"PU{plt_x},{plt_y};\n")
                    for j in range(37):
                        angle = j * 10 * math.pi / 180
                        circle_x = int(plt_x + radius * math.cos(angle))
                        circle_y = int(plt_y + radius * math.sin(angle))
                        if j == 0:
                            plt_file.write(f"PD{circle_x},{circle_y};\n")
                        else:
                            plt_file.write(f"{circle_x},{circle_y};\n")
                    plt_file.write("PU;\n")  # 抬笔
                    pbar.update(1)
                plt_file.flush()
        plt_file.write("SP0;\n")  # 放下笔
        plt_file.write("IN;\n")  # 结束

    print(f"PLT 文件数据写入完成: {filename}")
    print("正在完成文件保存...")
    for _ in tqdm(range(100), desc="最终处理"):
        time.sleep(0.05)  # 模拟最后的处理时间
    print(f"PLT 文件已成功保存: {filename}")

def save_svg(dots, width, height, filename):
    print(f"正在准备保存 SVG 文件: {filename}")
    dwg = svgwrite.Drawing(filename, size=(width, height))
    total_dots = len(dots)
    batch_size = 10000  # 每批处理的圆点数
    num_batches = total_dots // batch_size + (1 if total_dots % batch_size != 0 else 0)

    with tqdm(total=total_dots, desc="整体进度") as pbar:
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_dots)
            batch = dots[start:end]
            batch_desc = f"批次 {i + 1}/{num_batches}"
            for x, y, d in tqdm(batch, desc=batch_desc, leave=False):
                dwg.add(dwg.circle(center=(x, y), r=d / 2, fill='black'))
                pbar.update(1)
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                dwg.save()

    print(f"SVG 文件数据写入完成: {filename}")
    print("正在完成文件保存...")
    for _ in tqdm(range(100), desc="最终处理"):
        time.sleep(0.02)  # 模拟最后的处理时间
    print(f"SVG 文件已成功保存: {filename}")

def save_dxf(dots, filename):
    print(f"正在保存 DXF 文件: {filename}")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for x, y, d in tqdm(dots, desc="绘制 DXF"):
        msp.add_circle((x, y), d / 2)
    doc.saveas(filename)
    print(f"DXF 文件已保存: {filename}")

def save_png(dots, width, height, filename):
    print(f"正在保存 PNG 文件: {filename}")
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

    x, y, d = zip(*dots)
    e = EllipseCollection(
        widths=d, heights=d, angles=0,
        units='xy', offsets=list(zip(x, [height - yi for yi in y])),
        transOffset=ax.transData,
        facecolors='black'
    )
    ax.add_collection(e)

    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"PNG 文件已成功保存: {filename}")

def save_bmp(dots, width, height, filename):
    print(f"正在保存 BMP 文件: {filename}")
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    for x, y, d in tqdm(dots, desc="绘制 BMP"):
        bbox = (x - d/2, y - d/2, x + d/2, y + d/2)
        draw.ellipse(bbox, fill='black')

    image.save(filename, 'BMP')
    print(f"BMP 文件已成功保存: {filename}")

# 主程序
if __name__ == "__main__":
    image_path = '3.bmp'  # 请确保这里是您的BMP文件路径
    min_diameter = 0.1  # 最小圆直径（毫米）
    density_factor = 1.0  # 密度因子，增加会使圆点更稀疏
    threshold = 200  # 亮度阈值，只有高于此值的区域会添加圆点

    print("开始处理图像...")
    dots, width, height = image_to_dots(image_path, min_diameter, density_factor, threshold)

    # save_eps(dots, width, height, 'output.eps')
    # save_plt(dots, width, height, 'output.plt')
    # save_svg(dots, width, height, 'output.svg')
    # save_dxf(dots, 'output.dxf')
    save_png(dots, width, height, 'output.png')
    save_bmp(dots, width, height, 'output.bmp')  # 新增的BMP输出

    print("所有文件处理完成！")