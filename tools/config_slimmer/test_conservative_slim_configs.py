from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.config_slimmer.api_models import (
    API_MODEL_REGISTRY,
    modeled_api_names,
    resolve_api_model,
)
from tools.config_slimmer.conservative_slim_configs import (
    ConservativeOptions,
    _case_close,
    _case_distance,
    _sequence_sum_relative_delta,
    analyze_line,
    run,
    select_cases,
)


def _options(**overrides: object) -> ConservativeOptions:
    values: dict[str, object] = {
        "relative_tolerance": 0.02,
        "absolute_tolerance": 0.0,
        "max_absolute_delta": 256.0,
        "exact_small_integer": 16,
        "exact_integer_above": 1_000_000_000,
        "max_removal_rate": 0.20,
        "min_group_size": 2,
        "max_candidate_checks": 256,
        "seed": "test",
        "protect_boundaries": False,
    }
    values.update(overrides)
    return ConservativeOptions(**values)  # type: ignore[arg-type]


def _case(case_id: int, text: str):
    return analyze_line(case_id, Path("input.txt"), 0, case_id + 1, text)


class ApiModelRegistryTest(unittest.TestCase):
    def test_target_api_registry_is_explicit_and_complete(self) -> None:
        self.assertEqual(len(modeled_api_names()), 118)
        self.assertEqual(set(API_MODEL_REGISTRY), set(modeled_api_names()))
        self.assertEqual(resolve_api_model("paddle.transpose", None).name, "preserve")
        self.assertEqual(
            resolve_api_model("paddle.Tensor.cast", None).name,
            "cast_coverage",
        )
        self.assertEqual(
            resolve_api_model("paddle.concat", None).name,
            "preserve",
        )
        self.assertEqual(
            resolve_api_model("paddle.baddbmm", None).name,
            "linear_algebra_coverage",
        )
        self.assertEqual(
            resolve_api_model("paddle.nn.functional.moe_permute", None).name,
            "moe_permute",
        )

    def test_custom_and_unknown_models_are_resolved_safely(self) -> None:
        self.assertEqual(
            resolve_api_model(
                "paddle._C_ops._run_custom_op",
                "fuse_weighted_swiglu_fp8_quant",
            ).name,
            "custom_coverage",
        )
        self.assertEqual(
            resolve_api_model("paddle.unknown_future_api", None).name,
            "unmodeled_preserve",
        )


class AnalyzeTest(unittest.TestCase):
    def test_all_numeric_positions_are_parameterized(self) -> None:
        first = _case(
            0,
            'paddle.any_api(Tensor(paddle.Size([10000, 2048]),"float32"), axis=32, scale=1000.0, )',
        )
        second = _case(
            1,
            'paddle.any_api(Tensor(paddle.Size([10005, 2048]),"float32"), axis=32, scale=1005.0, )',
        )
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(len(first.numbers), 4)
        self.assertTrue(_case_close(first, second, _options()))

    def test_non_numeric_structure_must_match(self) -> None:
        first = _case(0, 'paddle.any_api(Tensor(paddle.Size([10000]),"float32"), flag=True, )')
        second = _case(1, 'paddle.any_api(Tensor(paddle.Size([10005]),"float16"), flag=True, )')
        third = _case(2, 'paddle.any_api(Tensor(paddle.Size([10005]),"float32"), flag=False, )')
        self.assertNotEqual(first.signature, second.signature)
        self.assertNotEqual(first.signature, third.signature)

    def test_string_whitespace_and_large_integers_remain_distinct(self) -> None:
        with_space = _case(0, 'paddle.any_api(9007199254740992, name="a b", )')
        without_space = _case(1, 'paddle.any_api(9007199254740992, name="ab", )')
        next_integer = _case(2, 'paddle.any_api(9007199254740993, name="a b", )')
        self.assertNotEqual(with_space.signature, without_space.signature)
        self.assertFalse(_case_close(with_space, next_integer, _options()))

    def test_moe_permute_profile_compares_token_distribution(self) -> None:
        first = _case(
            0,
            "paddle.nn.functional.moe_permute("
            'Tensor(paddle.Size([10000, 2048]),"bfloat16"), None, '
            'Tensor(paddle.Size([10000, 8]),"int32"), '
            'Tensor(paddle.Size([10000, 8]),"float32"), 8, '
            "list[1000,1200,1400,1600,1800,2000,2200,2400,], "
            "padding_alignment=128, do_gather=True, )",
        )
        second = _case(
            1,
            "paddle.nn.functional.moe_permute("
            'Tensor(paddle.Size([10005, 2048]),"bfloat16"), None, '
            'Tensor(paddle.Size([10005, 8]),"int32"), '
            'Tensor(paddle.Size([10005, 8]),"float32"), 8, '
            "list[2400,1800,1200,2200,1600,1000,2000,1400,], "
            "padding_alignment=128, do_gather=True, )",
        )
        self.assertEqual(first.numeric_sequences, ((7, 8, 9, 10, 11, 12, 13, 14),))
        self.assertTrue(_case_close(first, second, _options()))
        self.assertFalse(_case_close(first, second, _options(complex_api_profiles=False)))
        _, max_absolute, max_relative = _case_distance(first, second, _options())
        self.assertLessEqual(max_relative, 0.20)
        self.assertLessEqual(max_absolute, 1024)
        self.assertLessEqual(_sequence_sum_relative_delta(first, second, _options()), 0.05)

    def test_moe_permute_profile_rejects_distant_total_load(self) -> None:
        first = _case(
            0,
            "paddle.nn.functional.moe_permute(10000, 8, "
            "list[1000,1000,1000,1000,1000,1000,1000,1000,], )",
        )
        second = _case(
            1,
            "paddle.nn.functional.moe_permute(10005, 8, "
            "list[1100,1100,1100,1100,1100,1100,1100,1100,], )",
        )
        self.assertFalse(_case_close(first, second, _options()))

    def test_moe_unpermute_profile_preserves_linked_dimensions(self) -> None:
        first = _case(
            0,
            "paddle.nn.functional.moe_unpermute("
            'Tensor(paddle.Size([10000, 2048]),"bfloat16"), '
            'Tensor(paddle.Size([5000, 8]),"int32"), '
            'Tensor(paddle.Size([5000, 8]),"int32"), 5000, 8, )',
        )
        second = _case(
            1,
            "paddle.nn.functional.moe_unpermute("
            'Tensor(paddle.Size([10005, 2048]),"bfloat16"), '
            'Tensor(paddle.Size([5050, 8]),"int32"), '
            'Tensor(paddle.Size([5000, 8]),"int32"), 5050, 8, )',
        )
        self.assertFalse(_case_close(first, second, _options()))
        self.assertTrue(_case_close(first, second, _options(complex_api_profiles=False)))

    def test_one_distant_position_prevents_merge(self) -> None:
        first = _case(0, "paddle.any_api(10000, 20000, 30000, )")
        second = _case(1, "paddle.any_api(10005, 20010, 40000, )")
        self.assertFalse(_case_close(first, second, _options()))

    def test_small_integer_and_magnitude_boundaries_are_exact(self) -> None:
        self.assertFalse(
            _case_close(
                _case(0, "paddle.any_api(8, )"),
                _case(1, "paddle.any_api(9, )"),
                _options(),
            )
        )
        self.assertFalse(
            _case_close(
                _case(0, "paddle.any_api(1023, )"),
                _case(1, "paddle.any_api(1025, )"),
                _options(),
            )
        )


class SelectionTest(unittest.TestCase):
    def test_transpose_is_always_protected_as_a_complex_kernel(self) -> None:
        cases = [
            _case(
                index,
                "paddle.transpose("
                f'Tensor(paddle.Size([{10000 + index}, 2048]),"float32"), '
                "list[1,0,], )",
            )
            for index in range(100)
        ]
        select_cases(cases, _options())
        self.assertTrue(all(case.selected for case in cases))
        self.assertEqual({case.reason for case in cases}, {"preserve_model"})

    def test_simple_kernel_uses_coverage_selection_by_default(self) -> None:
        cases = [
            _case(
                index,
                f'paddle.empty(list[{10000 + index},2048,], dtype="float16", )',
            )
            for index in range(100)
        ]
        select_cases(cases, _options())
        removed = [case for case in cases if not case.selected]
        self.assertGreater(len(removed), 20)
        self.assertTrue(all(case.reason == "simple_coverage_sampled_out" for case in removed))

    def test_cast_has_a_higher_dedicated_retention_floor(self) -> None:
        cases = [
            _case(
                index,
                "paddle.Tensor.cast("
                f'Tensor(paddle.Size([{10000 + index}, 2048]),"float32"), '
                '"float16", )',
            )
            for index in range(100)
        ]
        select_cases(cases, _options())
        self.assertGreaterEqual(sum(case.selected for case in cases), 20)
        self.assertEqual({case.model_name for case in cases}, {"cast_coverage"})

    def test_data_rearrangement_and_indexing_apis_are_preserved(self) -> None:
        for api_name in ("paddle.concat", "paddle.Tensor.__getitem__"):
            cases = [_case(index, f"{api_name}({10000 + index}, 2048, )") for index in range(100)]
            select_cases(cases, _options())
            self.assertTrue(all(case.selected for case in cases), api_name)

    def test_modeled_near_api_supports_every_numeric_position(self) -> None:
        cases = [
            _case(index, f"paddle.Tensor.__eq__(32, {10000 + index}, 2048, )")
            for index in range(10)
        ]
        select_cases(cases, _options(max_removal_rate=0.30))
        removed = [case for case in cases if not case.selected]
        self.assertEqual(len(removed), 3)
        self.assertTrue(all(case.representative_id is not None for case in removed))
        self.assertTrue(all(case.differing_positions == (1,) for case in removed))

    def test_unmodeled_api_is_preserved(self) -> None:
        cases = [
            _case(index, f"paddle.unknown_future_api({10000 + index}, )") for index in range(100)
        ]
        select_cases(cases, _options())
        self.assertTrue(all(case.selected for case in cases))
        self.assertEqual({case.model_name for case in cases}, {"unmodeled_preserve"})

    def test_removal_rate_is_a_hard_cap(self) -> None:
        cases = [_case(index, f"paddle.Tensor.__eq__({10000 + index}, )") for index in range(100)]
        select_cases(cases, _options(max_removal_rate=0.07))
        self.assertEqual(sum(not case.selected for case in cases), 7)

    def test_similarity_does_not_chain_through_an_excluded_case(self) -> None:
        cases = [
            _case(0, "paddle.Tensor.__eq__(10000, )"),
            _case(1, "paddle.Tensor.__eq__(10150, )"),
            _case(2, "paddle.Tensor.__eq__(10300, )"),
        ]
        select_cases(cases, _options(max_removal_rate=1.0))
        self.assertTrue(cases[0].selected)
        self.assertFalse(cases[1].selected)
        self.assertTrue(cases[2].selected)
        self.assertEqual(cases[1].representative_id, cases[0].case_id)

    def test_boundary_protection_retains_extreme_representatives(self) -> None:
        cases = [_case(index, f"paddle.Tensor.__eq__({10000 + index}, )") for index in range(20)]
        select_cases(
            cases,
            _options(max_removal_rate=0.50, protect_boundaries=True),
        )
        by_value = {int(case.numbers[0].value): case for case in cases}
        self.assertTrue(by_value[10000].selected)
        self.assertTrue(by_value[10019].selected)


class EndToEndTest(unittest.TestCase):
    def test_run_writes_independent_outputs_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "configs.txt"
            lines = [f"paddle.Tensor.__eq__(32, {10000 + index}, 2048, )" for index in range(10)]
            source.write_text(
                "# retained comment\n\n" + "\n".join([*lines, lines[0]]) + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            args = argparse.Namespace(
                inputs=[source],
                output_dir=output,
                relative_tolerance=0.02,
                absolute_tolerance=0.0,
                max_absolute_delta=256.0,
                exact_small_integer=16,
                exact_integer_above=1_000_000_000,
                max_removal_rate=0.20,
                min_group_size=2,
                max_candidate_checks=256,
                seed="test",
                no_boundary_protection=True,
                no_complex_api_profiles=False,
                moe_sequence_relative_tolerance=0.20,
                moe_sequence_max_absolute_delta=1024.0,
                moe_sequence_sum_relative_tolerance=0.05,
                preserve_api=[],
                pin_file=[],
                preserve_input_order=True,
                dry_run=False,
                force=False,
            )
            report = run(args)

            slim = output / "configs_conservative_slim.txt"
            dedup = output / "configs_conservative_deduplicated.txt"
            excluded = output / "configs_conservative_excluded.txt"
            self.assertTrue(slim.is_file())
            self.assertTrue(dedup.is_file())
            self.assertTrue(excluded.is_file())
            self.assertTrue((output / "conservative_decisions.tsv").is_file())
            self.assertEqual(report["exact_duplicates_removed"], 1)
            self.assertEqual(report["excluded_near_duplicates"], 2)

            slim_configs = {
                line for line in slim.read_text().splitlines() if line and not line.startswith("#")
            }
            dedup_configs = {
                line for line in dedup.read_text().splitlines() if line and not line.startswith("#")
            }
            excluded_configs = set(excluded.read_text().splitlines())
            self.assertEqual(dedup_configs, slim_configs | excluded_configs)
            self.assertTrue(slim_configs.isdisjoint(excluded_configs))
            stored = json.loads((output / "conservative_report.json").read_text())
            self.assertEqual(stored["kept_cases"], report["kept_cases"])


if __name__ == "__main__":
    unittest.main()
