from __future__ import annotations

import pathlib
from typing import Any

from .spec import CompareSuite, ImplementationResult, PairwiseResult

DEFAULT_LARGE_ERROR_THRESHOLD = 1e-1
DEFAULT_CATEGORY_ORDER: dict[str, int] = {}
DEFAULT_IMPLEMENTATION_ORDER: dict[str, int] = {}
DEFAULT_DTYPE_ORDER = {"bf16": 0, "fp32": 1, "fp64": 2}
DEFAULT_CATEGORY_COLORS: dict[str, str] = {}


def fmt(value: float | str | None) -> str:
    if value in (None, ""):
        return "-"
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:.4e}"


def safe_name(text: str) -> str:
    return text.replace("/", "_").replace("|", "_").replace(" ", "_").replace(",", "_")


def metadata(result: ImplementationResult, key: str, default: Any = None) -> Any:
    return result.metadata.get(key, default)


def case_metadata(result: ImplementationResult, key: str, default: Any = None) -> Any:
    return result.metadata.get("case_metadata", {}).get(key, default)


def report_setting(suite: CompareSuite, key: str, default: Any = None) -> Any:
    return suite.report_config.get(key, default)


def shape_metadata_keys(suite: CompareSuite) -> list[str]:
    return list(report_setting(suite, "shape_metadata_keys", ["m", "k", "n"]))


def shape_tuple(result: ImplementationResult, suite: CompareSuite | None = None) -> tuple[Any, ...]:
    keys = shape_metadata_keys(suite) if suite is not None else ["m", "k", "n"]
    return tuple(case_metadata(result, key, 0) for key in keys)


def shape_label(shape: tuple[Any, ...]) -> str:
    return ",".join(str(v) for v in shape)


def shape_file_suffix(suite: CompareSuite, shape: tuple[Any, ...]) -> str:
    keys = shape_metadata_keys(suite)
    return "_".join(f"{key}{value}" for key, value in zip(keys, shape))


def ok_results(results: list[ImplementationResult]) -> list[ImplementationResult]:
    return [
        item for item in results if item.status == "ok" and item.metrics_vs_standard is not None
    ]


def result_sort_key(
    result: ImplementationResult, suite: CompareSuite | None = None
) -> tuple[int, int, int, int, int, str]:
    category = str(metadata(result, "category", ""))
    implementation = str(metadata(result, "implementation", result.spec.id))
    input_dtype = str(metadata(result, "input_dtype", result.spec.dtype))
    dweight_dtype = str(metadata(result, "dweight_dtype", result.spec.dtype))
    category_order = (
        report_setting(suite, "category_order", DEFAULT_CATEGORY_ORDER)
        if suite
        else DEFAULT_CATEGORY_ORDER
    )
    implementation_order = (
        report_setting(suite, "implementation_order", DEFAULT_IMPLEMENTATION_ORDER)
        if suite
        else DEFAULT_IMPLEMENTATION_ORDER
    )
    dtype_order = (
        report_setting(suite, "dtype_order", DEFAULT_DTYPE_ORDER) if suite else DEFAULT_DTYPE_ORDER
    )
    return (
        category_order.get(category, 99),
        implementation_order.get(implementation, 99),
        dtype_order.get(input_dtype, 99),
        dtype_order.get(dweight_dtype, 99),
        int(bool(result.spec.multi_precision)),
        result.spec.id,
    )


def metric_value(result: ImplementationResult, key: str) -> float:
    metric = result.metrics_vs_standard
    return float(getattr(metric, key)) if metric is not None else 0.0


def maybe_bold(text: str, result: ImplementationResult, suite: CompareSuite | None = None) -> str:
    threshold = (
        report_setting(suite, "large_error_threshold", DEFAULT_LARGE_ERROR_THRESHOLD)
        if suite
        else DEFAULT_LARGE_ERROR_THRESHOLD
    )
    return f"**{text}**" if metric_value(result, "max_abs") >= threshold else text


def implementation_config(result: ImplementationResult) -> str:
    return (
        f"`{metadata(result, 'implementation', result.spec.id)}` "
        f"(dtype={metadata(result, 'input_dtype', result.spec.dtype)}, "
        f"dweight={metadata(result, 'dweight_dtype', result.spec.dtype)}, "
        f"mp{1 if result.spec.multi_precision else 0})"
    )


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def positive_floor(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 1e-12
    return max(min(positives) / 10, 1e-12)


def plot_heatmaps(
    out_dir: pathlib.Path, suite: CompareSuite, pairwise_results: list[PairwiseResult]
) -> tuple[list[pathlib.Path], str | None]:
    try:
        plt = import_matplotlib()
        import numpy as np
        from matplotlib.colors import LogNorm
    except Exception as err:
        return [], f"{type(err).__name__}: {err}"

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    paths: list[pathlib.Path] = []
    case_ids = sorted({item.case_id for item in pairwise_results})
    reference_ids = suite.report_config.get("reference_order") or sorted(
        {item.expect.spec.id for item in pairwise_results}
    )
    target_ids = suite.report_config.get("target_order") or sorted(
        {item.actual.spec.id for item in pairwise_results}
    )

    for case_id in case_ids:
        rows = [item for item in pairwise_results if item.case_id == case_id]
        matrix = np.zeros((len(target_ids), len(reference_ids)), dtype=float)
        for item in rows:
            if item.actual.spec.id in target_ids and item.expect.spec.id in reference_ids:
                matrix[
                    target_ids.index(item.actual.spec.id), reference_ids.index(item.expect.spec.id)
                ] = item.metrics.max_abs
        fig, ax = plt.subplots(
            figsize=(max(8.0, len(reference_ids) * 1.1), max(4.0, len(target_ids) * 0.6))
        )
        positive = matrix[matrix > 0]
        image = (
            ax.imshow(matrix, cmap="YlOrRd", norm=LogNorm(vmin=max(positive.min(), 1e-12)))
            if positive.size
            else ax.imshow(matrix, cmap="YlOrRd")
        )
        ax.set_xticks(range(len(reference_ids)))
        ax.set_xticklabels(reference_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(target_ids)))
        ax.set_yticklabels(target_ids, fontsize=8)
        ax.set_title(f"{suite.op_name} pairwise max_abs: {case_id}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("max_abs")
        fig.tight_layout()
        path = figures_dir / f"pairwise_heatmap_{safe_name(case_id)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths, None


def plot_vs_standard(
    out_dir: pathlib.Path,
    suite: CompareSuite,
    rows: list[ImplementationResult],
    shape: tuple[Any, ...],
) -> pathlib.Path | None:
    rows = sorted(ok_results(rows), key=lambda item: result_sort_key(item, suite))
    if not rows:
        return None
    plt = import_matplotlib()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    values = [metric_value(row, "max_abs") for row in rows]
    floor = positive_floor(values)
    plot_values = [value if value > 0 else floor for value in values]
    labels = [
        f"{metadata(row, 'implementation', row.spec.id)}\n{metadata(row, 'input_dtype', row.spec.dtype)}/{metadata(row, 'dweight_dtype', row.spec.dtype)} mp{1 if row.spec.multi_precision else 0}"
        for row in rows
    ]
    category_colors = report_setting(suite, "category_colors", DEFAULT_CATEGORY_COLORS)
    colors = [category_colors.get(str(metadata(row, "category", "")), "#7f7f7f") for row in rows]
    fig, ax = plt.subplots(figsize=(18.0, max(6.0, 0.82 * len(rows) + 2.5)), dpi=160)
    bars = ax.barh(
        range(len(labels)),
        plot_values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.7,
        alpha=0.88,
    )
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_xscale("log")
    threshold = report_setting(suite, "large_error_threshold", DEFAULT_LARGE_ERROR_THRESHOLD)
    standard_label = report_setting(suite, "vs_standard_ylabel", "vs standard")
    shape_prefix = report_setting(suite, "shape_label_prefix", "shape")
    ax.axvline(
        threshold, color="#b00020", linestyle="--", linewidth=1.2, label="large error threshold"
    )
    ax.set_xlabel(f"max_abs {standard_label}")
    ax.set_title(f"All implementations vs standard, {shape_prefix}={shape_label(shape)}")
    ax.grid(axis="x", which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    xmax = max(plot_values + [threshold])
    for bar, row, value in zip(bars, rows, values):
        ax.text(
            bar.get_width() * 1.08,
            bar.get_y() + bar.get_height() / 2,
            fmt(value),
            va="center",
            fontsize=8,
            fontweight="bold" if metric_value(row, "max_abs") >= threshold else "normal",
        )
    ax.set_xlim(left=floor / 2, right=max(xmax * 8, threshold * 10))
    ax.legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.93, bottom=0.08)
    path = figures_dir / f"vs_standard_overview_{shape_file_suffix(suite, shape)}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def plot_reduce_trend(
    out_dir: pathlib.Path,
    suite: CompareSuite,
    results: list[ImplementationResult],
    metric_name: str,
) -> pathlib.Path | None:
    rows = ok_results(results)
    shapes = sorted({shape_tuple(row, suite) for row in rows})
    if len(shapes) < 2:
        return None
    plt = import_matplotlib()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160)
    groups: dict[str, list[ImplementationResult]] = {}
    for row in rows:
        trend_implementations = set(report_setting(suite, "reduce_trend_implementations", []))
        if not trend_implementations or metadata(row, "implementation") in trend_implementations:
            key = f"{metadata(row, 'implementation')} {metadata(row, 'input_dtype')}/{metadata(row, 'dweight_dtype')} mp{1 if row.spec.multi_precision else 0}"
            groups.setdefault(key, []).append(row)
    plotted = False
    for label, group_rows in groups.items():
        by_shape = {shape_tuple(row, suite): row for row in group_rows}
        x_key = report_setting(suite, "reduce_trend_x_metadata_key", shape_metadata_keys(suite)[0])
        points = [
            (
                case_metadata(by_shape[shape], x_key, shape[0]),
                metric_value(by_shape[shape], metric_name),
            )
            for shape in shapes
            if shape in by_shape
        ]
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plot_ys = [value if value > 0 else positive_floor(ys) for value in ys]
        is_fused = label.startswith("paddle_fused")
        ax.plot(
            xs,
            plot_ys,
            marker="D" if is_fused else "o",
            linewidth=3.0 if is_fused else 1.6,
            markersize=6.5 if is_fused else 4.2,
            label=label,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(report_setting(suite, "reduce_trend_x_label", "shape"))
    ax.set_ylabel(f"{metric_name} {report_setting(suite, 'vs_standard_ylabel', 'vs standard')}")
    ax.set_title(f"Reduce trend: M vs {metric_name}")
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    path = figures_dir / f"reduce_trend_{metric_name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def write_vs_standard_table(
    lines: list[str], suite: CompareSuite, rows: list[ImplementationResult]
) -> None:
    lines.append(
        "| 类型 | 实现 | input dtype | dweight dtype | multi precision | output dtype | max_abs | rmse | p99_abs | max_rel | p99_rel |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(ok_results(rows), key=lambda item: result_sort_key(item, suite)):
        metric = row.metrics_vs_standard
        assert metric is not None
        lines.append(
            f"| `{metadata(row, 'category')}` | `{metadata(row, 'implementation', row.spec.id)}` | "
            f"`{metadata(row, 'input_dtype', row.spec.dtype)}` | `{metadata(row, 'dweight_dtype', row.spec.dtype)}` | "
            f"`{row.spec.multi_precision}` | `{row.output_dtype}` | {maybe_bold(fmt(metric.max_abs), row, suite)} | "
            f"{maybe_bold(fmt(metric.rmse), row, suite)} | {maybe_bold(fmt(metric.p99_abs), row, suite)} | "
            f"{maybe_bold(fmt(metric.max_rel), row, suite)} | {maybe_bold(fmt(metric.p99_rel), row, suite)} |"
        )


def write_bitwise_table(
    lines: list[str],
    suite: CompareSuite,
    fused_rows: list[ImplementationResult],
    comparison_rows: list[ImplementationResult],
) -> None:
    lines.append("| Paddle fused 配置 | 逐位一致实现 | 逐位不一致实现 |")
    lines.append("| --- | --- | --- |")
    for fused in sorted(ok_results(fused_rows), key=lambda item: result_sort_key(item, suite)):
        fused_fp = metadata(fused, "output_fingerprint")
        comparable = [
            row
            for row in ok_results(comparison_rows)
            if metadata(row, "input_dtype") == metadata(fused, "input_dtype")
            and row.spec.multi_precision == fused.spec.multi_precision
        ]
        identical = [
            implementation_config(row)
            for row in comparable
            if fused_fp and metadata(row, "output_fingerprint") == fused_fp
        ]
        different = [
            implementation_config(row)
            for row in comparable
            if metadata(row, "output_fingerprint")
            and metadata(row, "output_fingerprint") != fused_fp
        ]
        lines.append(
            f"| {implementation_config(fused)} | {'；'.join(identical) if identical else '-'} | {'；'.join(different) if different else '-'} |"
        )


def write_profile_section(lines: list[str], profile: dict[str, Any] | None) -> None:
    lines.append("## 6. Kernel profile")
    lines.append("")
    if not profile:
        lines.append("未启用 kernel profile。")
        return
    if profile.get("status") == "skipped":
        lines.append(f"未采集 kernel 信息：`{profile.get('reason')}`。")
        if profile.get("manual_command_template"):
            lines.append("")
            lines.append("可手动使用 nsys 包裹如下 workload 命令：")
            lines.append("")
            lines.append("```bash")
            lines.append(str(profile["manual_command_template"]))
            lines.append("```")
        return
    rows = profile.get("kernel_summary") or []
    if not rows:
        lines.append(
            f"未解析到 kernel summary；profile 状态为 `{profile.get('status')}`。可查看 `profile/profile_status.json` 和原始 nsys/sqlite 产物。"
        )
        runs = profile.get("runs") or []
        if runs:
            lines.append("")
            lines.append("| implementation | status | detail |")
            lines.append("| --- | --- | --- |")
            for run in runs:
                detail = (
                    run.get("stderr")
                    or run.get("export_stderr")
                    or run.get("stdout")
                    or run.get("export_stdout")
                    or "-"
                )
                detail = str(detail).replace("|", "\\|").replace("\n", " ")[:240]
                lines.append(
                    f"| `{run.get('implementation')}` | `{run.get('status')}` | {detail} |"
                )
        return
    lines.append("| implementation | kernel | count | total ns | mean ns |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in rows[:30]:
        lines.append(
            f"| `{row.get('implementation')}` | `{row.get('kernel_name')}` | {row.get('count')} | {fmt(row.get('total_time_ns'))} | {fmt(row.get('mean_time_ns'))} |"
        )


def render_report(out_dir: pathlib.Path, run_data: dict[str, Any]) -> pathlib.Path:
    suite: CompareSuite = run_data["suite"]
    pairwise_results: list[PairwiseResult] = run_data["pairwise_results"]
    reference_pairwise_results: list[PairwiseResult] = run_data["reference_pairwise_results"]
    results: list[ImplementationResult] = run_data["results"]

    figure_paths, figure_error = plot_heatmaps(out_dir, suite, pairwise_results)
    vs_standard_error = None
    vs_standard_paths: list[pathlib.Path] = []
    reduce_paths: list[pathlib.Path] = []
    try:
        for shape in sorted({shape_tuple(row, suite) for row in results}):
            path = plot_vs_standard(
                out_dir, suite, [row for row in results if shape_tuple(row, suite) == shape], shape
            )
            if path:
                vs_standard_paths.append(path)
        for metric_name in ["max_abs", "rmse"]:
            path = plot_reduce_trend(out_dir, suite, results, metric_name)
            if path:
                reduce_paths.append(path)
    except Exception as err:
        vs_standard_error = f"{type(err).__name__}: {err}"

    global_max = (
        max(pairwise_results, key=lambda item: item.metrics.max_abs) if pairwise_results else None
    )
    reference_max = (
        max(reference_pairwise_results, key=lambda item: item.metrics.max_abs)
        if reference_pairwise_results
        else None
    )

    lines = [f"# {report_setting(suite, 'title', suite.op_name + ' 精度对比报告')}", ""]
    lines.append("## 1. 测试方法")
    lines.append("")
    lines.append(report_setting(suite, "method_intro", "测试对象为 operator compare suite。"))
    formula = report_setting(suite, "formula")
    if formula:
        lines.append("")
        lines.append("```text")
        lines.append(str(formula))
        lines.append("```")
    lines.append("")
    lines.append(f"- 标准实现：`{suite.standard_id}`。")
    lines.append(f"- 指标 dtype：`{suite.metrics_dtype}`。")
    lines.append("- TF32：关闭。")
    for key, value in suite.metadata.items():
        lines.append(f"- `{key}`：`{value}`。")

    lines.append("")
    lines.append("## 2. 实现列表")
    lines.append("")
    lines.append("| id | 名称 | group | category | dtype | dweight dtype | multi_precision |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for spec in suite.implementations:
        lines.append(
            f"| `{spec.id}` | {spec.display_name} | `{spec.group}` | `{spec.metadata.get('category')}` | `{spec.metadata.get('input_dtype', spec.dtype)}` | `{spec.metadata.get('dweight_dtype', spec.dtype)}` | `{spec.multi_precision}` |"
        )

    lines.append("")
    lines.append(f"## 3. {report_setting(suite, 'vs_standard_title', '相对标准实现误差')}")
    lines.append("")
    if vs_standard_error:
        lines.append(f"图形生成失败：`{vs_standard_error}`。")
        lines.append("")
    if reduce_paths:
        lines.append("### 3.1 跨 shape reduce 趋势")
        lines.append("")
        for path in reduce_paths:
            lines.append(f"![{path.stem}]({path.relative_to(out_dir)})")
            lines.append("")
    for index, shape in enumerate(sorted({shape_tuple(row, suite) for row in results}), 2):
        shape_rows = [row for row in results if shape_tuple(row, suite) == shape]
        lines.append(
            f"### 3.{index} {report_setting(suite, 'shape_label_prefix', 'shape')} = {shape_label(shape)}"
        )
        lines.append("")
        image = next(
            (path for path in vs_standard_paths if shape_file_suffix(suite, shape) in path.name),
            None,
        )
        if image:
            lines.append(f"![全部实现误差总览]({image.relative_to(out_dir)})")
            lines.append("")
        write_vs_standard_table(lines, suite, shape_rows)
        lines.append("")
        lines.append(f"#### {report_setting(suite, 'bitwise_title', '逐位一致性')}")
        lines.append("")
        primary_category = report_setting(suite, "bitwise_primary_category")
        comparison_categories = set(report_setting(suite, "bitwise_comparison_categories", []))
        write_bitwise_table(
            lines,
            suite,
            [row for row in shape_rows if metadata(row, "category") == primary_category],
            [row for row in shape_rows if metadata(row, "category") in comparison_categories],
        )
        lines.append("")

    lines.append("## 4. Pairwise 矩阵")
    lines.append("")
    if figure_error:
        lines.append(f"未生成 heatmap：`{figure_error}`。")
    elif figure_paths:
        for path in figure_paths:
            lines.append(f"![{path.stem}]({path.relative_to(out_dir)})")
            lines.append("")
    else:
        lines.append("没有可绘制的 pairwise 数据。")
    if global_max:
        lines.append(
            f"- 全局最大目标实现 vs 参考实现偏差：`max_abs={fmt(global_max.metrics.max_abs)}`，case `{global_max.case_id}`，`{global_max.actual.spec.id}` vs `{global_max.expect.spec.id}`。"
        )
    if reference_max:
        lines.append(
            f"- 参考实现之间最大直接偏差：`max_abs={fmt(reference_max.metrics.max_abs)}`，case `{reference_max.case_id}`，`{reference_max.actual.spec.id}` vs `{reference_max.expect.spec.id}`。"
        )

    lines.append("")
    lines.append("## 5. 失败实现")
    lines.append("")
    failed = [result for result in results if result.status != "ok"]
    if not failed:
        lines.append("无失败实现。")
    else:
        lines.append("| case | category | id | error |")
        lines.append("| --- | --- | --- | --- |")
        for result in failed:
            error = (result.error or "").replace("|", "\\|")
            lines.append(
                f"| `{result.case_id}` | `{metadata(result, 'category')}` | `{result.spec.id}` | {error} |"
            )

    lines.append("")
    write_profile_section(lines, run_data.get("profile"))

    lines.append("")
    lines.append("## 7. 结论")
    lines.append("")
    conclusions = []
    for category, title in report_setting(suite, "conclusion_categories", []):
        rows = [row for row in ok_results(results) if metadata(row, "category") == category]
        if rows:
            row = max(rows, key=lambda item: metric_value(item, "max_abs"))
            conclusions.append(
                f"{title} 相对最高标准的最大偏差为 `max_abs={fmt(metric_value(row, 'max_abs'))}`，配置为 {implementation_config(row)}。"
            )
    exact = [row for row in ok_results(results) if metric_value(row, "max_abs") == 0]
    if exact:
        exact_text = "；".join(implementation_config(row) for row in exact[:8])
        if len(exact) > 8:
            exact_text += f"；等 {len(exact)} 个配置"
        conclusions.append(f"与最高标准完全一致的配置包括：{exact_text}。")
    conclusions.append(
        "详细数据见 `summary.csv`、`pairwise_summary.csv`、`reference_pairwise_summary.csv`、`results.json` 和 `env.json`。"
    )
    for index, conclusion in enumerate(conclusions, 1):
        lines.append(f"{index}. {conclusion}")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
