from __future__ import annotations

for i in range(20):
    test_file = f"tester/api_config/api_config_merged_{i + 1}.txt"
    test_log = f"test_pipline/cpu_accuracy/cpu_accuracy_{i + 1}.log"
    with open(test_file) as file:
        lines = file.readlines()
        case_count = len(lines)

    with open(test_log) as file:
        content = file.read()
        test_count = content.count("test begin")
        pass_count = content.count("[Pass]")

    print(
        i + 1,
        "\t",
        case_count,
        "\t",
        test_count,
        "\t",
        case_count - test_count,
        "\t",
        f"{test_count * 100 / case_count:.2f}%",
        "\t",
        pass_count,
    )
