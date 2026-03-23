#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Run all evaluators in eval/5_tasks with one shared prompt file.
# Example:
#   bash run_all_5_tasks.sh --model gpt --video_path ./video_task1 --prompt_file ./prompt_obj.json --output_root ./runs

MODEL="gpt"
VIDEO_PATH="./video_task1"
PROMPT_FILE="./prompt_obj.json"
OUTPUT_ROOT="./runs"
NUM_WORKERS="4"
API_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --video_path)
      VIDEO_PATH="$2"
      shift 2
      ;;
    --prompt_file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --api_key)
      API_KEY="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$API_KEY" ]]; then
  if [[ "$MODEL" == "qwen" ]]; then
    API_KEY="${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-}}"
  else
    API_KEY="${OPENAI_API_KEY:-}"
  fi
fi

TASK_SCRIPTS=(
  "common_manipulation.py"
  "long-horizon_planning.py"
  "multi-entity_collaboration.py"
  "spatial_relationship.py"
  "visual_reasoning.py"
)

mkdir -p "$OUTPUT_ROOT"

for script in "${TASK_SCRIPTS[@]}"; do
  task_name="${script%.py}"
  out_dir="$OUTPUT_ROOT/$task_name/$MODEL"
  mkdir -p "$out_dir"

  echo "===== Running $script (model=$MODEL) ====="

  cmd=(
    python3 "$script"
    --video_path "$VIDEO_PATH"
    --read_prompt_file "$PROMPT_FILE"
    --output_path "$out_dir"
    --model "$MODEL"
    --num_workers "$NUM_WORKERS"
  )

  if [[ -n "$API_KEY" ]]; then
    cmd+=(--api_key "$API_KEY")
  fi

  "${cmd[@]}"
done

echo "All 5 tasks finished. Results saved under: $OUTPUT_ROOT"
echo "===== Summarizing final score (model=$MODEL) ====="
python3 summarize_5_tasks.py --output_root "$OUTPUT_ROOT" --model "$MODEL"
