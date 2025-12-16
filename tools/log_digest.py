from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys


def remove_time_stamp(file_path):
    """Process a log file by removing timestamp before 'test begin:' but keeping the 'test begin:' prefix.
    Writes the processed content back to the same file.

    Args:
        file_path (str): Path to the log file to process
    """
    try:
        # Read the file content
        with open(file_path) as file:
            lines = file.readlines()

        # Process each line
        processed_lines = []
        test_begin_count = 0
        for line in lines:
            # Skip grep warning line
            if "grep: warning: GREP_OPTIONS is deprecated" in line:
                continue

            # Check if the line contains "test begin: " and remove only the timestamp
            if "test begin: " in line:
                processed_line = re.sub(r"^.*?test begin: ", "test begin: ", line)
                processed_lines.append(processed_line)
                test_begin_count += 1
            else:
                processed_lines.append(line)

        # Write the processed content back to the file
        with open(file_path, "w") as file:
            file.writelines(processed_lines)

        print(
            f"Successfully processed {file_path} - processed {test_begin_count} 'time stamps' occurrences"
        )
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {e}", file=sys.stderr)
        sys.exit(1)


def extract_api_name(config_line):
    """Extract API name from configuration line (everything before the first '(')"""
    # Remove "test begin: " prefix if present
    if config_line.startswith("test begin: "):
        config_line = config_line[len("test begin: ") :]

    match = re.match(r"([^(]+)\(", config_line.strip())
    if match:
        return match.group(1).strip()
    return None


def process_log_entries(file_path, id, ckpt_id, write_pass, write_pass_amp):
    """Process log file by categorizing entries based on their status and distributing
    them to appropriate files.

    Args:
        file_path (str): Path to the log file to process
        id (str): Identifier to use in output filenames
        ckpt_id (str): Identifier to use in checkpoint file
        write_pass (bool): Whether to write pass api config to file
        write_pass_amp (bool): Whether to write pass api config to amp file
    """
    try:
        # First remove timestamps from the file
        remove_time_stamp(file_path)

        # Now read the processed file
        with open(file_path) as file:
            content = file.read()

        # Remove grep warning if present
        content = re.sub(r"grep: warning: GREP_OPTIONS is deprecated.*\n", "", content)

        # Split the content by "test begin: " to get individual test entries
        entries = re.split(r"test begin: ", content)

        # Skip the first entry if it's empty (before the first "test begin:")
        if entries and not entries[0].strip():
            entries = entries[1:]

        # Initialize counters
        pass_count = 0
        accuracy_error_count = 0
        paddle_error_count = 0
        torch_error_count = 0
        cuda_error_count = 0
        other_error_count = 0

        written_warned_log_files = []
        written_warned_config_files = []

        # Process each entry
        for entry in entries:
            # add \n to the end of the entry
            entry = entry + "\n"

            if not entry.strip():
                continue

            # Extract configuration (first line)
            config_line = entry.split("\n")[0].strip()
            api_name = extract_api_name(config_line)

            if not api_name:
                continue

            # Determine the status of the test
            if "[Pass]" in entry:
                # Write config to accuracy support file if passing write_pass
                support2torch_file = f"tester/api_config/api_config_support2torch{id}.txt"
                support2torch_amp_file = f"tester/api_config/api_config_support2torch_amp{id}.txt"
                if write_pass:
                    with open(support2torch_file, "a") as f:
                        f.write(f"{config_line}\n")
                if write_pass_amp:
                    with open(support2torch_amp_file, "a") as f:
                        f.write(f"{config_line}\n")
                pass_count += 1

            else:
                # Write log to log file
                error_type = (
                    "accuracy"
                    if "[accuracy error]" in entry
                    else "paddle"
                    if "[paddle error]" in entry
                    else "torch"
                    if "[torch error]" in entry
                    else "cuda"
                    if "[cuda error]" in entry
                    else "other"
                )
                log_or_cfg_dir = f"tester/api_config/test_log/{error_type}_error"
                os.makedirs(log_or_cfg_dir, exist_ok=True)
                log_file = f"{log_or_cfg_dir}/test_log_{api_name}.log"
                if not os.path.exists(log_file):
                    with open(log_file, "w") as f:
                        f.write(f"{entry}")
                    written_warned_log_files.append(log_file)
                else:
                    if log_file not in written_warned_log_files:
                        print(f"[warn] log file {log_file} already exists, appending to it")
                        written_warned_log_files.append(log_file)
                        with open(log_file, "a") as f:
                            f.write(
                                "\n===================== below is the new log =====================\n"
                            )
                    with open(log_file, "a") as f:
                        f.write(f"{entry}")

                # Write config to config file
                config_file = f"{log_or_cfg_dir}/{error_type}_error_api_config{id}.txt"
                if not os.path.exists(config_file):
                    with open(config_file, "w") as f:
                        f.write(f"{config_line}\n")
                    written_warned_config_files.append(config_file)
                else:
                    if config_file not in written_warned_config_files:
                        print(f"[warn] config file {config_file} already exists, appending to it")
                        written_warned_config_files.append(config_file)
                        with open(config_file, "a") as f:
                            f.write(
                                "\n===================== below is the new config =====================\n"
                            )
                    with open(config_file, "a") as f:
                        f.write(f"{config_line}\n")

                if error_type == "accuracy":
                    accuracy_error_count += 1
                elif error_type == "paddle":
                    paddle_error_count += 1
                elif error_type == "torch":
                    torch_error_count += 1
                elif error_type == "cuda":
                    cuda_error_count += 1
                elif error_type == "other":
                    other_error_count += 1

        # Read checkpoint file and compare counts
        checkpoint_file = f"tester/api_config/test_log/checkpoint{ckpt_id}.txt"
        checkpoint_count = 0
        try:
            with open(checkpoint_file) as f:
                checkpoint_lines = f.readlines()
                checkpoint_count = len(checkpoint_lines)
        except FileNotFoundError:
            print(f"[warn] checkpoint file {checkpoint_file} not found.")

        total_processed = (
            pass_count
            + accuracy_error_count
            + paddle_error_count
            + torch_error_count
            + cuda_error_count
            + other_error_count
        )

        print(f"Successfully processed and categorized log entries from {file_path}")
        print(f"Pass count:             {pass_count}")
        print(f"Accuracy error count:   {accuracy_error_count}")
        print(f"Paddle error count:     {paddle_error_count}")
        print(f"Torch error count:      {torch_error_count}")
        print(f"Cuda error count:       {cuda_error_count}")
        print(f"Other error count:      {other_error_count}")
        print(f"Total processed:        {total_processed}")
        print(f"Checkpoint count:       {checkpoint_count}")

        if checkpoint_count != total_processed:
            print(
                f"[warn] checkpoint count ({checkpoint_count}) does not match total processed count ({total_processed}), missing {total_processed - checkpoint_count} logs"
            )
        else:
            print("[info] checkpoint count matches total processed count.")

        return (
            pass_count,
            accuracy_error_count,
            paddle_error_count,
            torch_error_count,
            cuda_error_count,
            other_error_count,
        )

    except Exception as e:
        print(f"Error processing log entries: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process log files by removing timestamp and prefix. Ensure you run this script from the PaddleAPITest directory (cwd == PaddleAPITest)."
    )
    parser.add_argument(
        "--file",
        required=False,
        help='log file path to process, eg. if you run engine.py by " ... engine.py ... 2&>1 > log.log", then file=log.log',
    )
    parser.add_argument("--dir", required=False, help="Directory containing log files to process")
    parser.add_argument(
        "--id",
        default="",
        type=str,
        help="Identifier to use in output filenames, eg. lhy id==1, xym id==2",
    )
    parser.add_argument(
        "--ckpt-id",
        default="",
        type=str,
        help="Identifier to use in checkpoint file, usually ignore, using tester/api_config/test_log/checkpoint.txt",
    )
    parser.add_argument(
        "--write-pass",
        default=False,
        action="store_true",
        help="write pass api config to file",
    )
    parser.add_argument(
        "--write-pass-amp",
        default=False,
        action="store_true",
        help="write passed case to amp file",
    )

    args = parser.parse_args()
    args.id = "_" + args.id if args.id else ""
    args.ckpt_id = "_" + args.ckpt_id if args.ckpt_id else ""
    assert not (args.write_pass and args.write_pass_amp), (
        "write_pass and write_pass_amp cannot be set at the same time"
    )

    if args.file:
        # Check if error directories exist
        error_dirs = glob.glob("tester/api_config/test_log/*_error")
        if error_dirs:
            question = (
                f"Found error directories: {', '.join(error_dirs)}. Do you want to delete them?"
            )
            answer = input(f"{question} (y/n): ").strip().lower()
            if answer in ["y", "yes"]:
                for dir_path in error_dirs:
                    shutil.rmtree(dir_path)
                print(f"[info] Deleted error directories: {', '.join(error_dirs)}")

        process_log_entries(args.file, args.id, args.ckpt_id, args.write_pass, args.write_pass_amp)
        if not args.write_pass and not args.write_pass_amp:
            print(
                "\n[warn] --write-pass and --write-pass-amp are not set, passed api config will not be written to file"
            )
            print(
                "[hint] pass --write-pass or --write-pass-amp to write passed api config to file for final step after you fix all errors"
            )
        else:
            if args.write_pass:
                print(
                    f"\n[info] passed api config written to file tester/api_config/api_config_support2torch{args.id}.txt"
                )
            elif args.write_pass_amp:
                print(
                    f"\n[info] passed api config written to file tester/api_config/api_config_support2torch_amp{args.id}.txt"
                )
    elif args.dir:
        # TODO: add directory processing for parsing array of log ids and ckpt-ids
        raise NotImplementedError("Directory processing not implemented yet")
        # Process all files in the directory
        total_pass = 0
        total_accuracy_error = 0
        total_paddle_error = 0
        total_torch_error = 0

        for filename in os.listdir(args.dir):
            file_path = os.path.join(args.dir, filename)
            if os.path.isfile(file_path):
                print(f"\nProcessing file: {file_path}")
                (
                    pass_count,
                    accuracy_error_count,
                    paddle_error_count,
                    torch_error_count,
                ) = process_log_entries(file_path, args.id, args.ckpt_id)

                total_pass += pass_count
                total_accuracy_error += accuracy_error_count
                total_paddle_error += paddle_error_count
                total_torch_error += torch_error_count

        print("\nSummary for all files:")
        print(f"Total pass count: {total_pass}")
        print(f"Total accuracy error count: {total_accuracy_error}")
        print(f"Total paddle error count: {total_paddle_error}")
        print(f"Total torch error count: {total_torch_error}")
        print(
            f"Total processed: {total_pass + total_accuracy_error + total_paddle_error + total_torch_error}"
        )
    else:
        print("Error: Either --file or --dir must be specified", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
