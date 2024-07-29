import os

def count_lines_in_file(file_path):
    """统计单个文件的行数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)

def count_lines_in_folder(folder_path):
    """统计文件夹中所有 .txt 文件的总行数"""
    total_lines = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                lines = count_lines_in_file(file_path)
                total_lines += lines
                print(f"{file}: {lines} lines")
    return total_lines

if __name__ == "__main__":
    folder_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl_clip_0.3_0.8/'
    total_lines = count_lines_in_folder(folder_path)
    print(f"文件夹 '{folder_path}' 中所有 .txt 文件的总行数为: {total_lines}")
