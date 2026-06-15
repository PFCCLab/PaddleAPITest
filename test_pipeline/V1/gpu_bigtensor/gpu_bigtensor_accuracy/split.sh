#!/bin/bash

input_file="gpu_bigtensor_accuracy_errorconfig.txt"
output_prefix="gpu_bigtensor_accuracy_errorconfig_"
num_splits=8

# 获取文件总行数
total_lines=$(wc -l < "$input_file")
lines_per_split=$((total_lines / num_splits))
remainder=$((total_lines % num_splits))

current_line=0
split_index=1

# 打开输入文件
exec 3< "$input_file"

while IFS= read -r line || [[ -n "$line" ]]; do
    output_file="${output_prefix}${split_index}.txt"
    echo "$line" >> "$output_file"
    ((current_line++))

    # 当达到每份的行数时，切换到下一个输出文件
    if (( current_line == lines_per_split + (split_index <= remainder ? 1 : 0) )); then
        ((split_index++))
        current_line=0
    fi
done <&3

# 关闭输入文件描述符
exec 3<&-