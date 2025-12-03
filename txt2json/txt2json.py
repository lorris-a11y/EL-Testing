import os
import json

# 获取当前文件的父目录路径
current_directory = os.path.dirname(__file__)

# 定义文件路径（父目录中的 datasets 文件夹）
input_file = os.path.join(current_directory, '..', 'datasets', 'agnews_texts.txt')
output_file = os.path.join(current_directory, 'agNews.json')

# 读取文本文件
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建一个列表，每个元素包含一个 "original" 键，值为文本中的一行
    data = [{"original": line.strip()} for line in lines if line.strip()]

    # 将数据写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Data has been written to {output_file}")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
