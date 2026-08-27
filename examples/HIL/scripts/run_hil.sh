#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HIL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${HIL_LOG_DIR:-${HIL_DIR}/logs}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-29}"
VR_POSE_URI="${VR_POSE_URI:-ws://127.0.0.1:8080/devicepose}"
VR_POSE_TRANSPORT="${VR_POSE_TRANSPORT:-zenoh}"
VR_ZENOH_ENDPOINT="${VR_ZENOH_ENDPOINT:-}"
VR_ZENOH_KEY="${VR_ZENOH_KEY:-sec/xr/devicepose}"
HIL_ZENOH_SITE_PACKAGES="${HIL_ZENOH_SITE_PACKAGES:-}"
POLICY_CONTROL_HZ="${HIL_POLICY_CONTROL_HZ:-15}"
POLICY_SYNC_SLOP="${HIL_POLICY_SYNC_SLOP:-0.02}"
PYTHON_BIN="${HIL_PYTHON_BIN:-python3}"
HIL_ENABLE_API_OUTPUT="${HIL_ENABLE_API_OUTPUT:-0}"
ROS_LOG_DIR="${ROS_LOG_DIR:-${LOG_DIR}/ros}"
HIL_PYTHONPYCACHEPREFIX="${HIL_PYTHONPYCACHEPREFIX:-/tmp/hil-python-${UID}/pycache}"

mkdir -p "${LOG_DIR}" "${ROS_LOG_DIR}" "${HIL_PYTHONPYCACHEPREFIX}"
export PYTHONPATH="${HIL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_DOMAIN_ID ROS_LOG_DIR PYTHONPATH HIL_ZENOH_SITE_PACKAGES PYTHONPYCACHEPREFIX="${HIL_PYTHONPYCACHEPREFIX}"

# Use an isolated bytecode cache so a damaged system .pyc cannot prevent all
# control processes from starting. Import the common ROS path once before
# spawning children to fail with a concise message if the Python stack is bad.
if ! "${PYTHON_BIN}" -c 'import tempfile; import lark; import rclpy; from rcl_interfaces.msg import ParameterEvent'; then
  echo "[HIL] Python/ROS 启动检查失败：${PYTHON_BIN}" >&2
  exit 1
fi

echo "[HIL] 启动：ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, API_OUTPUT=${HIL_ENABLE_API_OUTPUT}"
echo "[HIL] VR：transport=${VR_POSE_TRANSPORT}, key=${VR_ZENOH_KEY}, endpoint=${VR_ZENOH_ENDPOINT:-auto-discovery}"
echo "[HIL] 等待 VR 位姿流和外部 policy；运行日志同时写入 ${LOG_DIR}"
echo "[HIL] 手柄：右 A=遥操，左 X=policy，右 B/左 Y=暂停锁停"

pids=()
shutdown_started=0
shutdown() {
  if [[ "${shutdown_started}" == "1" ]]; then
    return
  fi
  shutdown_started=1
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
stop_from_signal() {
  shutdown
  exit 130
}
trap shutdown EXIT
trap stop_from_signal INT TERM

"${PYTHON_BIN}" -u -m hil.policy_topic_adapter \
  --control-hz "${POLICY_CONTROL_HZ}" \
  --sync-slop "${POLICY_SYNC_SLOP}" \
  > >(tee "${LOG_DIR}/policy_topic_adapter.log") 2>&1 &
pids+=("$!")

vr_args=(
  --pose-transport "${VR_POSE_TRANSPORT}"
  --uri "${VR_POSE_URI}"
  --zenoh-key "${VR_ZENOH_KEY}"
)
if [[ -n "${VR_ZENOH_ENDPOINT}" ]]; then
  vr_args+=(--zenoh-endpoint "${VR_ZENOH_ENDPOINT}")
fi

"${PYTHON_BIN}" -u -m hil.vr_teleop_server "${vr_args[@]}" \
  > >(tee "${LOG_DIR}/vr_teleop_server.log") 2>&1 &
pids+=("$!")

# Start the only /api publisher last. External policy publishes four input
# topics below /hil/policy/input; the adapter combines them into one chunk.
supervisor_args=()
if [[ "${HIL_ENABLE_API_OUTPUT}" != "1" ]]; then
  supervisor_args+=(--dry-run)
fi
"${PYTHON_BIN}" -u -m hil.hil_supervisor "${supervisor_args[@]}" &
pids+=("$!")

# Exit and tear down the full control group if any required process dies.
set +e
wait -n "${pids[@]}"
status=$?
set -e
echo "A required HIL process exited (status=${status}); check ${LOG_DIR}/*.log" >&2
exit 1
