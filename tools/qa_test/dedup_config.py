#!/usr/bin/env python3
"""Deduplicate configuration lines while preserving sorted unique output.

Usage:
    python dedup_config.py -i api_config_0_size.txt
    python dedup_config.py -i api_config_0_size.txt -o /output/dir/dedup.txt
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate configuration lines while preserving sorted unique output.",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input config file path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path. Default: <input_stem>_dedup.txt in same dir as input")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_dedup{ext}"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    seen = set()
    total = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            seen.add(line.rstrip("\n"))

    unique_lines = sorted(seen)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    print(f"Total lines:   {total}")
    print(f"Unique lines:  {len(unique_lines)}")
    print(f"Duplicates:    {total - len(unique_lines)}")
    print(f"Output:        {output_path}")


if __name__ == "__main__":
    main()
