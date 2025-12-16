from __future__ import annotations

import argparse
import os

path = "/root/PaddleAPITest/tester/api_config/5_accuracy/"
keyword = "accuracy"


def getnum(dst):
    s = set()
    with open(path + dst) as f:
        data = f.readlines()
        for j in data:
            tmp = j.split("(")[0]
            if tmp not in s:
                s.add(tmp)
    return len(s)


def isclean(config):
    list = os.listdir(path)
    merge_list = []
    for i in list:
        if i.find(keyword) != -1:
            merge_list.append(i)

    cnt = 0
    exist_list = set()
    exist_nums = {}
    for i in merge_list:
        cnt2 = 0
        with open(path + i, encoding="utf8") as f:
            data = f.readlines()
            for j in data:
                if config in j:
                    cnt += 1
                    cnt2 += 1
                    exist_list.add(i)
        exist_nums[i] = cnt2

    if cnt:
        print(config[: len(config) - 1] + " is still exist, number of times : ", cnt)
        print(config[: len(config) - 1] + " is still exist in these files : ")
        for i in exist_list:
            print(i, exist_nums[i])
        return 0
    else:
        print("clean")
        return 1


def add(config, dst):
    list = os.listdir(path)
    merge_list = []
    for i in list:
        if i.find(keyword) != -1:
            merge_list.append(i)

    for i in merge_list:
        with open(path + i) as f:
            lines = f.readlines()

        matched_lines = [line for line in lines if config in line]

        with open(path + dst, "a+") as f:
            f.writelines(matched_lines)

        if len(matched_lines):
            print("add", len(matched_lines), "lines in", i, "to ", dst, "successfully")


def remove(config):
    list = os.listdir(path)
    merge_list = []
    for i in list:
        if i.find(keyword) != -1:
            merge_list.append(i)

    for i in merge_list:
        with open(path + i) as f:
            lines = f.readlines()

        count = sum(config in line for line in lines)
        remaining_lines = [line for line in lines if config not in line]

        with open(path + i, "w") as f:
            f.writelines(remaining_lines)

        if count:
            print("remove", count, "lines in", i, "successfully")


def main():
    parser = argparse.ArgumentParser(description="Process config and temporary file.")
    parser.add_argument("--config", type=str, required=True, help="配置字符串，例如 paddle.numel")
    parser.add_argument("--dst", type=str, default="mytmp.txt", help="临时文件名，默认是 mytmp.txt")
    parser.add_argument("--remove", action="store_true", help="是否执行删除配置")
    args = parser.parse_args()

    config = args.config
    dst = args.dst
    print(f"开始处理配置：{config}，目标文件：{dst}")
    config += "("

    if not isclean(config):
        add(config, dst)  # 向指定临时文件以a+方式添加写入，不会改变原有配置

    if args.remove:
        print(f"执行删除配置：{config[: len(config) - 1]}")
        remove(config)  # 仅当设置了--remove时才执行删除


if __name__ == "__main__":
    main()
