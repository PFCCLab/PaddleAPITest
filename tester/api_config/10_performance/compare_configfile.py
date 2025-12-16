# 定义读取文件的函数
from __future__ import annotations


def read_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        # 返回文件中的所有行，并去掉行尾的换行符
        return {line.strip() for line in file}


# 定义写入文件的函数
def write_to_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for line in data:
            file.write(line + "\n")


def filter_strings(test_file, pass_file, result_file):
    # 读取两个文件中的字符串
    test_strings = read_file(test_file)
    pass_strings = read_file(pass_file)

    # 找出test.txt中有，pass.txt中没有的字符串
    result_strings = test_strings - pass_strings

    # 将结果写入新的文件
    write_to_file(result_file, result_strings)
    print(f"筛选完成，结果已保存到 {result_file}")


# 使用示例
pass_file = "pass.txt"  # 输入文件test.txt路径
test_file = "error.txt"  # 输入文件pass.txt路径
result_file = "re_error_config.txt"  # 输出文件result.txt路径

filter_strings(test_file, pass_file, result_file)
