from __future__ import annotations

from pathlib import Path

from tester.api_config.parser import APIConfig
from tools.regression import collect_configs


def test_iter_source_files_selects_4096_and_1m_and_excludes_unstable_paths(
    tmp_path: Path,
):
    regular_4096 = tmp_path / "model" / "accuracy" / "2048_4096_8192.txt"
    regular_1m = tmp_path / "model" / "accuracy" / "1M.txt"
    fp8_1m = tmp_path / "model" / "accuracy" / "1M_fp8_extended.txt"
    need_fix = tmp_path / "model" / "accuracy" / "2048_4096_8192_need_fix.txt"
    needfix = tmp_path / "model" / "accuracy" / "1M_needfix.txt"
    not_monitor = tmp_path / "model" / "accuracy" / "1M_getitem_not_monitor.txt"
    ignored = tmp_path / "model" / "accuracy" / "0size.txt"
    for path in (
        regular_4096,
        regular_1m,
        fp8_1m,
        need_fix,
        needfix,
        not_monitor,
        ignored,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    assert list(collect_configs.iter_source_files([tmp_path])) == [
        regular_4096,
        regular_1m,
        fp8_1m,
    ]


def test_collect_configs_keeps_all_4096_apis_and_float8(tmp_path: Path):
    source = tmp_path / "2048_4096_8192.txt"
    source.write_text(
        'paddle.unsupported(Tensor([2, 3],"float8_e4m3fn"), )\n',
        encoding="utf-8",
    )

    selected, stats = collect_configs.collect_configs([tmp_path], max_per_api=5)

    assert selected["paddle.unsupported"] == [
        'paddle.unsupported(Tensor([2, 3],"float8_e4m3fn"), )'
    ]
    assert stats["selected"] == 1


def test_collect_configs_keys_custom_ops_by_first_argument(tmp_path: Path):
    source = tmp_path / "4096.txt"
    source.write_text(
        "\n".join(
            [
                'paddle._C_ops._run_custom_op("op_a", Tensor([2, 3],"float32"), )',
                'paddle._C_ops._run_custom_op("op_b", Tensor([2, 3],"float32"), )',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected, _ = collect_configs.collect_configs([tmp_path], max_per_api=5)

    assert set(selected) == {
        "paddle._C_ops._run_custom_op:op_a",
        "paddle._C_ops._run_custom_op:op_b",
    }
    assert all(len(configs) == 1 for configs in selected.values())


def test_collect_configs_uses_smallest_1m_shapes_to_fill_4096_bucket(tmp_path: Path):
    (tmp_path / "4096.txt").write_text(
        'paddle.add(Tensor([4, 4],"float32"), Tensor([4, 4],"float32"), )\n',
        encoding="utf-8",
    )
    (tmp_path / "1M.txt").write_text(
        "\n".join(
            f'paddle.add(Tensor([{size}],"float32"), Tensor([{size}],"float32"), )'
            for size in (100, 5, 2, 4, 3)
        )
        + "\n",
        encoding="utf-8",
    )

    selected, _ = collect_configs.collect_configs([tmp_path], max_per_api=5)

    assert selected["paddle.add"] == [
        'paddle.add(Tensor([4, 4],"float32"), Tensor([4, 4],"float32"), )',
        'paddle.add(Tensor([2],"float32"), Tensor([2],"float32"), )',
        'paddle.add(Tensor([3],"float32"), Tensor([3],"float32"), )',
        'paddle.add(Tensor([4],"float32"), Tensor([4],"float32"), )',
        'paddle.add(Tensor([5],"float32"), Tensor([5],"float32"), )',
    ]


def test_collect_configs_selects_smallest_shapes_for_1m_only_api(tmp_path: Path):
    (tmp_path / "1M.txt").write_text(
        "\n".join(f'paddle.new_api(Tensor([{size}],"float32"), )' for size in (8, 1, 7, 2, 6, 3))
        + "\n",
        encoding="utf-8",
    )

    selected, _ = collect_configs.collect_configs([tmp_path], max_per_api=5)

    assert [config.args[0].shape for config in map(APIConfig, selected["paddle.new_api"])] == [
        [1],
        [2],
        [3],
        [6],
        [7],
    ]


def test_collect_configs_sorts_tensorless_shape_arguments_numerically(tmp_path: Path):
    (tmp_path / "1M.txt").write_text(
        "\n".join(f"paddle.ones(list[{size},], )" for size in (9, 10, 8, 2, 7, 3)) + "\n",
        encoding="utf-8",
    )

    selected, _ = collect_configs.collect_configs([tmp_path], max_per_api=5)

    assert [config.args[0] for config in map(APIConfig, selected["paddle.ones"])] == [
        [2],
        [3],
        [7],
        [8],
        [9],
    ]
