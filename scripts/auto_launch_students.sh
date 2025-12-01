#!/usr/bin/env bash
set -euo pipefail

# Wait until a teacher checkpoint is ready, then launch student jobs on GPU 2–6
# Usage: scripts/auto_launch_students.sh

TEACHER_CKPT="paper_results/fakeddit6_time/fakeddit6_teacher_roberta_time_seed01/model_best.pt"
STUDENTS_JOBS="scripts/jobs_fakeddit6_time_students.txt"
STUDENTS_STAGE1="scripts/jobs_fakeddit6_time_students_stage1.txt"
LOCK_DIR="tmp/auto_students"
mkdir -p "$LOCK_DIR"

echo "[auto] Waiting for teacher checkpoint: $TEACHER_CKPT"
until [[ -f "$TEACHER_CKPT" ]]; do
  sleep 60
done
echo "[auto] Detected teacher checkpoint. Launching student batches."

if [[ -f "$STUDENTS_JOBS" ]]; then
  if [[ ! -f "$LOCK_DIR/launched_students_main" ]]; then
    nohup bash scripts/run_with_free_gpus.sh "$STUDENTS_JOBS" 600 \
      > "$LOCK_DIR/students_main.out" 2>&1 &
    echo $! > "$LOCK_DIR/students_main.pid"
    touch "$LOCK_DIR/launched_students_main"
    echo "[auto] Launched main students (PID $(cat $LOCK_DIR/students_main.pid))."
  fi
fi

if [[ -f "$STUDENTS_STAGE1" ]]; then
  if [[ ! -f "$LOCK_DIR/launched_students_stage1" ]]; then
    nohup bash scripts/run_with_free_gpus.sh "$STUDENTS_STAGE1" 600 \
      > "$LOCK_DIR/students_stage1.out" 2>&1 &
    echo $! > "$LOCK_DIR/students_stage1.pid"
    touch "$LOCK_DIR/launched_students_stage1"
    echo "[auto] Launched stage1 students (PID $(cat $LOCK_DIR/students_stage1.pid))."
  fi
fi

echo "[auto] Done."

