from __future__ import annotations

for i in range(8):
    test_file = f"test_pipline/gpu_bigtensor/gpu_bigtensor_accuracy/gpu_bigtensor_accuracy_errorconfig_{i + 1}.txt"
    test_log = f"test_pipline/gpu_bigtensor/gpu_bigtensor_accuracy/gpu_bigtensor_accuracy_errorconfig_{i + 1}.log"
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
