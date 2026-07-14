#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_DIR="${ROOT_DIR}/.repro"
WS_DIR="${REPRO_DIR}/ros2_ws"
VENV_DIR="${REPRO_DIR}/venv"
CODA_ROOT_DIR="${CODA_ROOT_DIR:-/home/user/Code/coda-devkit/data}"
BAG_DIR="${CODA_ROOT_DIR}/rosbags/coda_0_with_depth_full_20260610_192234.bag"
QOS_FILE="${REPRO_DIR}/tf_overrides.yaml"

activate_runtime() {
  set +u
  source /opt/ros/jazzy/setup.bash
  source "${WS_DIR}/install/setup.bash"
  source "${VENV_DIR}/bin/activate"
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
  export ROS_STATIC_TRANSFORM_BROADCASTER_QOS_OVERRIDES="${QOS_FILE}"
  cd "${ROOT_DIR}"
}

check_runtime() {
  activate_runtime
  test -d "${BAG_DIR}"
  test -f checkpoints/fastsam/FastSAM-x-640x480.engine
  test -f checkpoints/reid_weights/clip_general.engine
  test -f config/labels_pseudo.yaml
  test -f config/labels_pseudo.csv
  python - <<'PY'
import cv2
import cvxpy
import hydra_python
import rclpy
import spark_dsg
import tensorrt
import torch

assert torch.cuda.is_available()
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"tensorrt={tensorrt.__version__}")
print(f"cv2={cv2.__version__}")
print(f"cvxpy_solvers={','.join(cvxpy.installed_solvers())}")
print("ROS/Hydra/Python imports: OK")
PY
  ros2 bag info "${BAG_DIR}"
}

launch_pipeline() {
  activate_runtime
  exec ros2 launch daaam_ros coda_daaam_hydra.launch.yaml \
    scene:="${SCENE_NAME:-coda_sequence_0}" \
    "$@"
}

play_bag() {
  activate_runtime
  local duration=""
  if [[ "${1:-}" == "--duration" ]]; then
    test -n "${2:-}"
    duration="$2"
    shift 2
  fi
  local command=(
    ros2 bag play "${BAG_DIR}"
    --clock
    --qos-profile-overrides-path "${QOS_FILE}"
    "$@"
  )
  if [[ -n "${duration}" ]]; then
    local status=0
    timeout --signal=INT --kill-after=30 "${duration}s" "${command[@]}" || status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 130 ]]; then
      return 0
    fi
    return "${status}"
  fi
  exec "${command[@]}"
}

postprocess_run() {
  test -n "${1:-}"
  local run_dir
  run_dir="$(realpath "$1")"
  activate_runtime
  python scripts/postprocess_scene_graph.py \
    --data-dir "${run_dir}" \
    --sentence-model-name sentence-transformers/sentence-t5-xl
  ros2 launch daaam_ros cluster_places.launch.yaml \
    data_dir:="${run_dir}" \
    interactive:=false
}

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_coda_demo.sh check
  scripts/reproduce_coda_demo.sh launch [ROS launch overrides...]
  scripts/reproduce_coda_demo.sh play [--duration SECONDS] [ros2 bag play args...]
  scripts/reproduce_coda_demo.sh postprocess OUTPUT_RUN_DIR

Run `launch` and `play` in separate terminals. Start `play` only after every
DAAAM worker reports ready. A 60-second smoke test uses `play --duration 60`.
EOF
}

case "${1:-}" in
  check)
    shift
    check_runtime "$@"
    ;;
  launch)
    shift
    launch_pipeline "$@"
    ;;
  play)
    shift
    play_bag "$@"
    ;;
  postprocess)
    shift
    postprocess_run "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
