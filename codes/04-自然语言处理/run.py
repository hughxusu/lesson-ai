# import os
# import re
# import jieba
# from tqdm import tqdm
#
#
# def clean_text(text):
#     """简单的文本清理函数"""
#     # 移除HTML标签
#     text = re.sub(r'<[^>]+>', '', text)
#     # 移除特殊字符
#     text = re.sub(r'[^\w\s]', '', text)
#     # 移除多余空格和换行符
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
#
# def process_file(input_file, output_file):
#     """处理单个文件，将其转换为FastText格式"""
#     try:
#         # 指定使用GBK编码读取文件
#         with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
#             content = f.read()
#     except UnicodeDecodeError:
#         print(f"警告: 文件 {input_file} 使用GBK解码失败，尝试使用UTF-8")
#         try:
#             with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
#                 content = f.read()
#         except Exception as e:
#             print(f"错误: 文件 {input_file} 无法解码: {e}")
#             return
#
#     # 简单地按段落分割文本
#     paragraphs = content.split('\n\n')
#
#     with open(output_file, 'a', encoding='utf-8') as out_f:
#         for paragraph in paragraphs:
#             if not paragraph.strip():
#                 continue
#
#             # 清理文本
#             cleaned_text = clean_text(paragraph)
#
#             # 使用jieba进行中文分词
#             words = jieba.cut(cleaned_text)
#
#             # 确保至少有一个词
#             words_list = list(words)
#             if len(words_list) > 0:
#                 # 将分词结果用空格连接
#                 line = ' '.join(words_list)
#                 # 写入输出文件
#                 out_f.write(line + '\n')
#
#
# def convert_sogou_to_fasttext(data_dir, output_file):
#     """将整个搜狐新闻数据集转换为FastText格式"""
#     # 确保输出文件为空
#     if os.path.exists(output_file):
#         os.remove(output_file)
#
#     # 获取所有文件列表
#     all_files = []
#     for root, dirs, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith('.txt'):
#                 all_files.append(os.path.join(root, file))
#
#     # 处理每个文件
#     print(f"找到 {len(all_files)} 个文件需要处理")
#     for file_path in tqdm(all_files, desc="处理文件"):
#         process_file(file_path, output_file)
#
#     print(f"转换完成，输出文件保存在: {output_file}")
#
#
# if __name__ == "__main__":
#     # 输入数据目录
#     data_dir = "./data/yule"
#
#     # 输出文件路径
#     output_file = "./data/THUCNews_yule.txt"
#
#     # 执行转换
#     convert_sogou_to_fasttext(data_dir, output_file)

import fasttext

# 输入文件路径（转换后的文本文件）
input_file = "./data/THUCNews_yule.txt"

# 输出模型路径
output_model = "./data/THUCNews_yule"

# 训练词向量模型
model = fasttext.train_unsupervised(
    input=input_file,
    model='skipgram',  # 可以是'skipgram'或'cbow'
    dim=300,           # 词向量维度
    ws=5,              # 上下文窗口大小
    epoch=5,           # 训练轮数
    minCount=5,        # 最小词频
)


model.save_model(f"{output_model}.bin")
