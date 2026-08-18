#!/usr/bin/env python3
"""Compare dense real risk/compute frontiers for ExtraTrees and depth-8 trees."""

import argparse
import json
from pathlib import Path


def dense_points(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text())
    points = []
    for name, metrics in report["aggregate"].items():
        if not name.startswith("dense_"):
            continue
        points.append({
            "name": name,
            "flops": metrics["theoretical_flops_saved_pct"],
            "harm5": 100.0 * metrics["instance_harm5_rate"],
            "mase_change": 100.0 * (metrics["geomean_cell_mase_ratio"] - 1.0),
            "coverage": 100.0 * metrics["coverage"],
        })
    return sorted(points, key=lambda row: row["name"])


def pareto(points: list[dict], include_mase: bool) -> list[dict]:
    result = []
    for point in points:
        dominated = False
        for other in points:
            no_worse = (
                other["flops"] >= point["flops"]
                and other["harm5"] <= point["harm5"]
            )
            if include_mase:
                no_worse &= other["mase_change"] <= point["mase_change"]
            strictly_better = (
                other["flops"] > point["flops"]
                or other["harm5"] < point["harm5"]
                or (include_mase and other["mase_change"] < point["mase_change"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(point)
    return result


def dominates(other: dict, point: dict, include_mase: bool) -> bool:
    no_worse = (
        other["flops"] >= point["flops"]
        and other["harm5"] <= point["harm5"]
    )
    if include_mase:
        no_worse &= other["mase_change"] <= point["mase_change"]
    strictly_better = (
        other["flops"] > point["flops"]
        or other["harm5"] < point["harm5"]
        or (include_mase and other["mase_change"] < point["mase_change"])
    )
    return bool(no_worse and strictly_better)


def cross_dominated(points: list[dict], alternatives: list[dict],
                    include_mase: bool) -> int:
    return sum(
        any(dominates(other, point, include_mase) for other in alternatives)
        for point in points
    )


def best_at_harm(points: list[dict], budget: float) -> dict | None:
    feasible = [point for point in points if point["harm5"] <= budget + 1e-12]
    return max(feasible, key=lambda row: row["flops"], default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    all_points = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        tree_path = model_dir / "distill_tree_depth8" / "real_evaluation.json"
        teacher_path = model_dir / "teacher_dense" / "real_evaluation.json"
        if not (tree_path.exists() and teacher_path.exists()):
            continue
        tree = dense_points(tree_path)
        teacher = dense_points(teacher_path)
        union = [dict(point, family="tree") for point in tree]
        union += [dict(point, family="extra_trees") for point in teacher]
        front2 = pareto(union, include_mase=False)
        front3 = pareto(union, include_mase=True)
        for point in union:
            all_points.append(dict(point, model=model_dir.name))
        row = {
            "model": model_dir.name,
            "tree_points_on_2d_union_front": sum(
                point["family"] == "tree" for point in front2),
            "teacher_points_on_2d_union_front": sum(
                point["family"] == "extra_trees" for point in front2),
            "tree_points_on_3d_union_front": sum(
                point["family"] == "tree" for point in front3),
            "teacher_points_on_3d_union_front": sum(
                point["family"] == "extra_trees" for point in front3),
            "nonnative_tree_points_on_2d_union_front": sum(
                point["family"] == "tree" and point["name"] != "dense_00"
                for point in front2),
            "nonnative_teacher_points_on_2d_union_front": sum(
                point["family"] == "extra_trees" and point["name"] != "dense_00"
                for point in front2),
            "nonnative_tree_points_on_3d_union_front": sum(
                point["family"] == "tree" and point["name"] != "dense_00"
                for point in front3),
            "nonnative_teacher_points_on_3d_union_front": sum(
                point["family"] == "extra_trees" and point["name"] != "dense_00"
                for point in front3),
            "teacher_points_2d_dominated_by_tree": cross_dominated(
                teacher[1:], tree[1:], include_mase=False),
            "tree_points_2d_dominated_by_teacher": cross_dominated(
                tree[1:], teacher[1:], include_mase=False),
            "teacher_points_3d_dominated_by_tree": cross_dominated(
                teacher[1:], tree[1:], include_mase=True),
            "tree_points_3d_dominated_by_teacher": cross_dominated(
                tree[1:], teacher[1:], include_mase=True),
            "tree_2d_front": [point for point in front2 if point["family"] == "tree"],
            "teacher_2d_front": [point for point in front2 if point["family"] == "extra_trees"],
            "tree_3d_front": [point for point in front3 if point["family"] == "tree"],
            "teacher_3d_front": [point for point in front3 if point["family"] == "extra_trees"],
            "harm_budgets": {},
        }
        for budget in (0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0):
            row["harm_budgets"][str(budget)] = {
                "tree": best_at_harm(tree, budget),
                "extra_trees": best_at_harm(teacher, budget),
            }
        rows.append(row)

    summary = {
        "models": rows,
        "n_models": len(rows),
        "models_with_tree_on_2d_front": sum(
            row["tree_points_on_2d_union_front"] > 0 for row in rows),
        "models_with_tree_on_3d_front": sum(
            row["tree_points_on_3d_union_front"] > 0 for row in rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
