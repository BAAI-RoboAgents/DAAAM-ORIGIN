#!/usr/bin/env python3
"""Create reproducible, read-only E13 comparison and review aids."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs/"
    "diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labeled_tile(path: Path, label: str, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size[0], 34), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def require_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    summaries = json.loads((root / "tables/variant_summary.json").read_text())
    action_rows = []
    proxy_rows = []
    review_rows = []
    representative_tiles = []
    collision_tiles = []
    for summary in summaries:
        identifier = summary["variant_id"]
        variant_root = root / "variants" / identifier
        events = read_jsonl(variant_root / "merge_events.jsonl")
        membership = json.loads((variant_root / "entity_membership.json").read_text())
        timelines = json.loads(
            (variant_root / "track_entity_timelines.json").read_text()
        )
        counts = Counter(event["action"] for event in events)
        for action_name in [
            "created_new",
            "new_track_merged",
            "local_track_continued",
            "local_track_reassociated_new",
            "local_track_reassociated_existing",
        ]:
            action_rows.append(
                {
                    "variant_id": identifier,
                    "threshold_m": summary["threshold_m"],
                    "action": action_name,
                    "count": counts[action_name],
                    "fraction_of_observations": counts[action_name] / len(events),
                }
            )
        multi_count = sum(int(row["unique_track_count"] > 1) for row in membership)
        split_count = sum(int(row["unique_entity_count"] > 1) for row in timelines)
        proxy_rows.append(
            {
                "variant_id": identifier,
                "threshold_m": summary["threshold_m"],
                "entity_count": len(membership),
                "multi_track_entity_count_proxy": multi_count,
                "multi_track_entity_fraction_proxy": multi_count / len(membership),
                "same_frame_extra_track_collision_count_proxy": summary[
                    "same_frame_multi_track_collision_count_proxy"
                ],
                "split_or_reassociated_track_count_proxy": split_count,
                "split_or_reassociated_track_fraction_proxy": split_count
                / len(timelines),
                "entity_spread_exceeds_threshold_count_proxy": sum(
                    int(row["spread_exceeds_threshold_proxy"])
                    for row in membership
                ),
                "large_dimension_over_3m_entity_count_proxy": sum(
                    int(row["maximum_dimension_m"] > 3.0) for row in membership
                ),
                "maximum_entity_spread_m": summary[
                    "maximum_entity_observation_spread_m"
                ],
            }
        )
        per_entity_frame: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            per_entity_frame[
                (int(event["entity_ordinal"]), int(event["frame_index"]))
            ].append(event)
        candidates = []
        for member in membership:
            ordinal = int(member["entity_ordinal"])
            frame_candidates = []
            for (candidate_ordinal, frame_index), frame_events in per_entity_frame.items():
                if candidate_ordinal != ordinal:
                    continue
                tracks = sorted({int(event["track_id"]) for event in frame_events})
                frame_candidates.append((len(tracks), -frame_index, frame_events, tracks))
            maximum = max(frame_candidates, key=lambda item: (item[0], item[1]))
            candidates.append(
                (
                    maximum[0],
                    int(member["same_frame_multi_track_collision_count"]),
                    maximum[2],
                    maximum[3],
                    member,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (same_frame_tracks, aggregate_collisions, frame_events, tracks, member) in enumerate(
            candidates[:8], start=1
        ):
            representative = frame_events[0]
            row = {
                "schema": "daaam.g1_no_gt_e13_collision_review.v1",
                "variant_id": identifier,
                "threshold_m": summary["threshold_m"],
                "rank_within_variant": rank,
                "entity_ordinal": member["entity_ordinal"],
                "review_frame_index": representative["frame_index"],
                "review_source_frame_index": representative["source_frame_index"],
                "same_frame_unique_track_count": same_frame_tracks,
                "same_frame_track_ids_json": json.dumps(tracks),
                "aggregate_extra_track_collisions": aggregate_collisions,
                "entity_unique_track_count": member["unique_track_count"],
                "entity_observation_count": member["observation_count"],
                "maximum_dimension_m": member["maximum_dimension_m"],
                "maximum_observation_distance_to_final_center_m": member[
                    "maximum_observation_distance_to_final_center_m"
                ],
                "overlay_path": str(
                    (
                        variant_root
                        / "frames"
                        / f"{representative['frame_index']:08d}"
                        / "entity_overlay.jpg"
                    ).resolve()
                ),
                "correctness_label": None,
                "requires_human_review": True,
            }
            review_rows.append(row)
            if rank <= 3:
                collision_tiles.append(
                    labeled_tile(
                        Path(row["overlay_path"]),
                        (
                            f"{summary['threshold_m']:.2f}m E{member['entity_ordinal']} "
                            f"src={representative['source_frame_index']} "
                            f"tracks={tracks}"
                        ),
                    )
                )
        for frame_index in (0, 50, 100):
            representative_tiles.append(
                labeled_tile(
                    variant_root
                    / "frames"
                    / f"{frame_index:08d}"
                    / "entity_overlay.jpg",
                    (
                        f"{summary['threshold_m']:.2f}m "
                        f"frame={frame_index} source={frame_index + 473}"
                    ),
                )
            )
    write_json(root / "analysis/action_counts.json", action_rows)
    write_csv(root / "analysis/action_counts.csv", action_rows)
    write_json(root / "analysis/entity_proxy_summary.json", proxy_rows)
    write_csv(root / "analysis/entity_proxy_summary.csv", proxy_rows)
    write_jsonl(root / "analysis/top_collision_review.jsonl", review_rows)
    write_csv(root / "analysis/top_collision_review.csv", review_rows)
    require_image(
        root / "visualizations/03_representative_entity_overlays.jpg",
        np.vstack(
            [
                np.hstack(representative_tiles[index : index + 3])
                for index in range(0, len(representative_tiles), 3)
            ]
        ),
    )
    require_image(
        root / "visualizations/04_collision_review_gallery.jpg",
        np.vstack(
            [
                np.hstack(collision_tiles[index : index + 3])
                for index in range(0, len(collision_tiles), 3)
            ]
        ),
    )
    action_by_variant = {
        summary["variant_id"]: {
            row["action"]: row["count"]
            for row in action_rows
            if row["variant_id"] == summary["variant_id"]
        }
        for summary in summaries
    }
    lines = [
        "# E13 结果细化分析",
        "",
        "本页只解释结构代理，不给实体合并正确性或最佳门限。所有候选都使用同一组 "
        "2,886 条合格 3D 观察。",
        "",
        "## 动作与风险代理",
        "",
        "| 门限 | created new | new-track merged | continued | reassociated new/existing | multi-track entity 占比 | track 多实体占比 | 同帧额外 track 冲突 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary, proxy in zip(summaries, proxy_rows):
        actions = action_by_variant[summary["variant_id"]]
        lines.append(
            f"| {summary['threshold_m']:.2f} m | {actions['created_new']} | "
            f"{actions['new_track_merged']} | {actions['local_track_continued']} | "
            f"{actions['local_track_reassociated_new']}/"
            f"{actions['local_track_reassociated_existing']} | "
            f"{proxy['multi_track_entity_fraction_proxy']:.1%} | "
            f"{proxy['split_or_reassociated_track_fraction_proxy']:.1%} | "
            f"{proxy['same_frame_extra_track_collision_count_proxy']} |"
        )
    lines.extend(
        [
            "",
            "门限由 0.20 m 增到 0.50 m 时，entity 数和 local-track 多实体代理下降，"
            "说明更宽门限确实吸收了更多空间碎片；但 multi-track entity 占比从 "
            "14.5% 增到 27.0%，同帧额外 track 冲突从 177 增到 840。后者是强烈的"
            "人工复核信号：同一时刻独立 BotSort track 被投到一个实体，可能来自真实"
            "跟踪碎片，也可能来自 FastSAM 重叠/嵌套 mask 或误合并，当前无 GT 不能区分。",
            "",
            "0.20 m 也不能直接冻结：它产生 220 个 entity，51/101 个 track 曾跨多个 "
            "entity，且有 237 次 local-track 重关联，显示尺度/深度/位姿波动会使严格门限"
            "频繁拆分。0.35 m 位于两端之间，但仍有 581 个同帧额外 track 冲突；“折中”"
            "不是正确性的证据。",
            "",
            "## 复核入口",
            "",
            "- `top_collision_review.jsonl/csv` 给出每个门限前 8 个同帧冲突候选、"
            "对应 source frame、track 列表和 overlay。",
            "- `03_representative_entity_overlays.jpg` 固定比较 source 473/523/573。",
            "- `04_collision_review_gallery.jpg` 汇总每个门限最强的 3 个冲突候选。",
            "- 正确性字段保持为空；人工复核或 GT 到位前不改写为 over-merge。",
            "",
        ]
    )
    (root / "analysis/RESULT_INTERPRETATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "action_rows": len(action_rows),
                "proxy_rows": len(proxy_rows),
                "review_rows": len(review_rows),
                "visualizations": 2,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
