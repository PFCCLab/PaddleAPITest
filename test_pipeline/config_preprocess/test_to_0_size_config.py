"""验证配置行的顶层调用拆分协议。"""

# 测试同时覆盖共享结构规范化和 0-size 入口的 APIConfig 语义解析。
# 重点保证脏数据可定位、可落盘，并且不会阻止后续合法调用被处理。

import tempfile
import unittest
from pathlib import Path

from test_pipeline.config_preprocess.config_lines import split_top_level_calls
from test_pipeline.config_preprocess.normalize_config_lines import normalize_file
from test_pipeline.config_preprocess.to_0_size_config import iter_api_configs


class SplitTopLevelCallsTest(unittest.TestCase):
    """确保拆分不会改变独立调用的内容和顺序。"""

    def test_split_two_calls_without_separator(self):
        text = "paddle.zeros([1])paddle.randn([2])"
        self.assertEqual(
            split_top_level_calls(text),
            ["paddle.zeros([1])", "paddle.randn([2])"],
        )

    def test_nested_parentheses_and_string_are_not_split(self):
        # 字符串中的伪调用和右括号不能提前结束外层调用。
        text = 'paddle.foo((1, 2), "paddle.bar())")paddle.zeros([1])'
        self.assertEqual(
            split_top_level_calls(text),
            ['paddle.foo((1, 2), "paddle.bar())")', "paddle.zeros([1])"],
        )

    def test_unbalanced_or_trailing_text_is_rejected(self):
        # 无法无损拆分的数据必须失败，不能只保留可识别的前缀。
        with self.assertRaisesRegex(ValueError, "括号不匹配"):
            split_top_level_calls("paddle.zeros([1]")
        with self.assertRaisesRegex(ValueError, "无法识别"):
            split_top_level_calls("paddle.zeros([1]) + 1")

    def test_api_name_scan_does_not_cross_into_next_prefix(self):
        with self.assertRaisesRegex(ValueError, "缺少左括号"):
            split_top_level_calls("paddle.foopaddle.bar(1)")

    def test_iter_api_configs_parses_each_call(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as config_file:
            config_file.write("paddle.zeros([1])paddle.randn([2])\n")
            config_file.flush()
            configs = list(iter_api_configs(config_file.name))
        self.assertEqual(
            [config.api_name for config in configs],
            ["paddle.zeros", "paddle.randn"],
        )

    def test_api_config_error_contains_source_line(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as config_file:
            config_file.write("paddle.add(Tensor([1]), Tensor([1]))\n")
            config_file.flush()
            with self.assertRaisesRegex(
                ValueError, rf"{config_file.name}:1: 无法解析拆分后的调用"
            ):
                list(iter_api_configs(config_file.name))

    def test_api_config_reject_callback_continues_with_later_calls(self):
        rejected = []
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as config_file:
            config_file.write(
                "paddle.add(Tensor([1]), Tensor([1]))\n"
                "paddle.zeros([1])\n"
            )
            config_file.flush()
            configs = list(
                iter_api_configs(
                    config_file.name,
                    on_reject=lambda *details: rejected.append(details),
                )
            )

        self.assertEqual([config.api_name for config in configs], ["paddle.zeros"])
        self.assertEqual(rejected[0][1], 1)
        self.assertIn("无法解析拆分后的调用", str(rejected[0][3]))

    def test_normalize_file_keeps_valid_calls_and_records_rejects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.txt"
            output = Path(temp_dir) / "normalized.txt"
            rejects = Path(temp_dir) / "normalized.txt.unparsed.txt"
            source.write_text(
                "paddle.zeros([1])paddle.randn([2])\n"
                "paddle.foopaddle.bar(1)\n",
                encoding="utf-8",
            )
            counts = normalize_file(source, output, rejects)

            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["paddle.zeros([1])", "paddle.randn([2])"],
            )
            self.assertEqual(counts["split_lines"], 1)
            self.assertEqual(counts["rejected"], 1)
            self.assertIn(":2\t", rejects.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
