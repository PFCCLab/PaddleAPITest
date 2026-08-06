from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.config_slimmer.slim_configs import (
    SlimOptions,
    analyze_line,
    load_cases,
    run,
    select_cases,
)


def _options(**overrides: object) -> SlimOptions:
    values: dict[str, object] = {
        "rates": {"custom": 0.75, "high": 0.5, "medium": 0.25, "low": 0.1},
        "minimums": {"custom": 2, "high": 2, "medium": 1, "low": 1},
        "seed": "test",
    }
    values.update(overrides)
    return SlimOptions(**values)  # type: ignore[arg-type]


class AnalyzeLineTest(unittest.TestCase):
    def test_custom_op_is_high_priority_and_keeps_name(self) -> None:
        line = (
            'paddle._C_ops._run_custom_op("my_fused_op", '
            'Tensor(paddle.Size([64, 128]),"bfloat16"), True, )'
        )
        case = analyze_line(0, Path("input.txt"), 0, 1, line, _options())
        self.assertEqual(case.api_name, "paddle._C_ops._run_custom_op")
        self.assertEqual(case.custom_name, "my_fused_op")
        self.assertEqual(case.priority, "custom")
        self.assertIn("custom=my_fused_op", case.base_features)

    def test_simple_shape_variants_share_signature(self) -> None:
        first = 'paddle.transpose(Tensor(paddle.Size([63, 128]),"float32"), [1,0], )'
        second = 'paddle.transpose(Tensor(paddle.Size([65, 256]),"float32"), [1,0], )'
        a = analyze_line(0, Path("input.txt"), 0, 1, first, _options())
        b = analyze_line(1, Path("input.txt"), 0, 2, second, _options())
        self.assertEqual(a.signature, b.signature)
        self.assertEqual(a.priority, "low")

    def test_unparsed_lines_are_preserved(self) -> None:
        options = _options()
        cases = [
            analyze_line(index, Path("input.txt"), 0, index + 1, f"unknown {index}", options)
            for index in range(5)
        ]
        select_cases(cases, options)
        self.assertTrue(all(case.selected for case in cases))


class SelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_and_preserves_boundaries(self) -> None:
        options = _options(rates={"custom": 0.75, "high": 0.5, "medium": 0.25, "low": 0.2})
        texts = [
            f'paddle.transpose(Tensor(paddle.Size([{size}, 128]),"float32"), [1,0], )'
            for size in (63, 64, 65, 100, 101, 127, 128, 129, 200, 256)
        ]

        def selected_texts() -> set[str]:
            cases = [
                analyze_line(i, Path("input.txt"), 0, i + 1, text, options)
                for i, text in enumerate(texts)
            ]
            select_cases(cases, options)
            return {case.text for case in cases if case.selected}

        first = selected_texts()
        self.assertEqual(first, selected_texts())
        self.assertTrue(any("[63," in text for text in first))
        self.assertTrue(any("[256," in text for text in first))

    def test_exact_duplicates_are_removed_before_analysis(self) -> None:
        options = _options()
        text = 'paddle.zeros(list[4,4,], "float32", )'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text(f"{text}\n{text}\n", encoding="utf-8")
            cases, _, preprocessing = load_cases([source], options)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].duplicate_occurrences, 1)
        self.assertEqual(preprocessing["config_occurrences"], 2)
        self.assertEqual(preprocessing["exact_duplicates_removed"], 1)

    def test_slim_strength_reduces_retention(self) -> None:
        texts = [
            'paddle.transpose(Tensor(paddle.Size([16, 128]),"float32"), [1,0], )' + (" " * index)
            for index in range(20)
        ]
        aggressive = _options(slim_strength=0.0)
        default = _options(slim_strength=1.0)
        aggressive_cases = [
            analyze_line(i, Path("input.txt"), 0, i + 1, text, aggressive)
            for i, text in enumerate(texts)
        ]
        default_cases = [
            analyze_line(i, Path("input.txt"), 0, i + 1, text, default)
            for i, text in enumerate(texts)
        ]
        select_cases(aggressive_cases, aggressive)
        select_cases(default_cases, default)
        self.assertLess(
            sum(case.selected for case in aggressive_cases),
            sum(case.selected for case in default_cases),
        )


class EndToEndTest(unittest.TestCase):
    def test_run_writes_new_files_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "configs.txt"
            source.write_text(
                "# retained comment\n\n"
                + "".join(
                    f'paddle.transpose(Tensor(paddle.Size([{n}, 128]),"float32"), [1,0], )\n'
                    for n in range(16, 36)
                )
                + 'paddle.transpose(Tensor(paddle.Size([16, 128]),"float32"), [1,0], )\n',
                encoding="utf-8",
            )
            output = root / "output"
            args = argparse.Namespace(
                inputs=[source],
                output_dir=output,
                high_rate=0.5,
                custom_rate=0.75,
                medium_rate=0.25,
                low_rate=0.1,
                min_high=2,
                min_custom=2,
                min_medium=1,
                min_low=1,
                high_api=[],
                custom_api=[],
                medium_api=[],
                low_api=[],
                preserve_api=[],
                pin_file=[],
                seed="test",
                slim_strength=1.0,
                preserve_input_order=False,
                progress=False,
                dry_run=False,
                force=False,
            )
            report = run(args)
            slimmed = output / "configs_slim.txt"
            deduplicated = output / "configs_deduplicated.txt"
            excluded = output / "configs_excluded.txt"
            self.assertTrue(slimmed.is_file())
            self.assertLess(len(slimmed.read_text().splitlines()), 20)
            slimmed_lines = slimmed.read_text().splitlines()
            self.assertEqual(slimmed_lines, sorted(slimmed_lines))
            self.assertIn("# retained comment", slimmed.read_text().splitlines())
            deduplicated_configs = {
                line
                for line in deduplicated.read_text().splitlines()
                if line and not line.startswith("#")
            }
            slimmed_configs = {
                line
                for line in slimmed.read_text().splitlines()
                if line and not line.startswith("#")
            }
            excluded_configs = set(excluded.read_text().splitlines())
            self.assertEqual(deduplicated_configs, slimmed_configs | excluded_configs)
            self.assertTrue(slimmed_configs.isdisjoint(excluded_configs))
            stored_report = json.loads((output / "coverage_report.json").read_text())
            self.assertEqual(stored_report["kept_cases"], report["kept_cases"])
            self.assertEqual(stored_report["exact_duplicates_removed"], 1)
            self.assertTrue((output / "decisions.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
