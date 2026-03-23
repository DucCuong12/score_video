import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize 5 task results into one final score")
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--model", required=True, type=str, choices=["gpt", "qwen"])
    args = parser.parse_args()

    task_dirs = sorted(
        [d for d in os.listdir(args.output_root) if os.path.isdir(os.path.join(args.output_root, d))]
    )

    rows = []
    all_scores = []

    for task in task_dirs:
        csv_path = os.path.join(args.output_root, task, args.model, "results.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "score" not in df.columns:
            continue

        scores = pd.to_numeric(df["score"], errors="coerce")
        scores = scores[scores >= 0]
        if scores.empty:
            task_mean = None
            count = 0
        else:
            task_mean = float(scores.mean())
            count = int(scores.shape[0])
            all_scores.extend(scores.tolist())

        rows.append(
            {
                "task": task,
                "num_samples": count,
                "mean_score_raw": round(task_mean, 4) if task_mean is not None else None,
                "mean_score_norm_0_1": round((task_mean - 1) / 4.0, 4) if task_mean is not None else None,
            }
        )

    if not rows:
        print("No valid results found to summarize.")
        return

    summary_df = pd.DataFrame(rows)

    final_raw = None
    final_norm = None
    if all_scores:
        final_raw = float(sum(all_scores) / len(all_scores))
        final_norm = (final_raw - 1) / 4.0

    final_row = {
        "task": "ALL_TASKS_FINAL",
        "num_samples": len(all_scores),
        "mean_score_raw": round(final_raw, 4) if final_raw is not None else None,
        "mean_score_norm_0_1": round(final_norm, 4) if final_norm is not None else None,
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([final_row])], ignore_index=True)

    out_csv = os.path.join(args.output_root, f"final_summary_{args.model}.csv")
    summary_df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
