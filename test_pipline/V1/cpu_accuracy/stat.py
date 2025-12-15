for i in range(20):
    test_file = "tester/api_config/api_config_merged_{}.txt".format(i + 1)
    test_log = "test_pipline/cpu_accuracy/cpu_accuracy_{}.log".format(i + 1)
    with open(test_file, "r") as file:
        lines = file.readlines()
        case_count = len(lines)

    with open(test_log, "r") as file:
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
        "{:.2f}%".format((test_count * 100 / case_count)),
        "\t",
        pass_count,
    )
