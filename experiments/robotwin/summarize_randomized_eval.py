import argparse
import csv
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"


def load_tasks(task_file: Path) -> list[str]:
    with task_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid task file: {task_file}")
    return list(dict.fromkeys(data.keys()))


def parse_result(path: Path) -> float | None:
    if not path.exists():
        return None
    value = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            continue
    return value


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--target-episodes", default=100, type=int)
    parser.add_argument("--task-file", default=TASK_FILE, type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    tasks = load_tasks(args.task_file)
    rows = []
    completed_rates = []
    weighted_success = 0
    weighted_episodes = 0

    for task in tasks:
        task_root = run_root / "tasks" / task
        progress = load_progress(task_root / "progress_random.json")
        result_rate = parse_result(task_root / task / "_result_random.txt")

        completed = int(progress.get("completed_episodes", 0) or 0)
        success_count = progress.get("success_count")
        if success_count is not None:
            success_count = int(success_count)

        if result_rate is not None and completed >= args.target_episodes:
            success_rate = float(result_rate)
            completed = args.target_episodes
            if success_count is None:
                success_count = int(round(success_rate * args.target_episodes))
            status = "done"
        elif completed > 0:
            success_rate = float(success_count / completed) if success_count is not None else None
            status = "partial"
        else:
            success_rate = None
            status = "pending"

        if status == "done" and success_rate is not None:
            completed_rates.append(success_rate)
        if completed > 0 and success_count is not None:
            weighted_success += success_count
            weighted_episodes += completed

        rows.append(
            {
                "task_name": task,
                "status": status,
                "completed_episodes": completed,
                "target_episodes": args.target_episodes,
                "success_count": success_count,
                "random_success_rate": success_rate,
                "next_seed": progress.get("next_seed"),
                "updated_at": progress.get("updated_at"),
            }
        )

    done_count = sum(1 for row in rows if row["status"] == "done")
    payload = {
        "run_root": str(run_root),
        "target_episodes_per_task": args.target_episodes,
        "num_tasks": len(rows),
        "done_tasks": done_count,
        "partial_or_done_episodes": weighted_episodes,
        "overall": {
            "mean_task_success_rate_done_only": (
                float(sum(completed_rates) / len(completed_rates)) if completed_rates else None
            ),
            "weighted_success_rate_partial_or_done": (
                float(weighted_success / weighted_episodes) if weighted_episodes else None
            ),
        },
        "per_task": rows,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary_randomized.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_root / "summary_randomized.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["overall"], ensure_ascii=False))
    print(f"done_tasks={done_count}/{len(rows)} partial_or_done_episodes={weighted_episodes}")


if __name__ == "__main__":
    main()
