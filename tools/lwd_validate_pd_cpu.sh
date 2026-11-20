#!/usr/bin/env bash
# Run the hardware-independent PD migration contracts in both source trees.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ascend_dir="${LWD_ASCEND_SOURCE:-${repo_dir}/../vllm-ascend}"
python_bin="${LWD_PD_PYTHON:-${repo_dir}/.venv-lwd-cpu/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Set LWD_PD_PYTHON to the Python executable in the uv-managed CPU test environment." >&2
  exit 2
fi
cd "${repo_dir}"
"${python_bin}" -m pytest -q --confcutdir=tests/v1/lwd_control \
  tests/v1/lwd_control/test_lwd_pd_protocol.py \
  tests/v1/lwd_control/test_lwd_pd_edge_contracts.py \
  tests/v1/lwd_control/test_lwd_pd_cloud_contracts.py \
  tests/v1/lwd_control/test_lwd_pd_proxy.py \
  tests/v1/lwd_control/test_lwd_frontend_result.py
"${python_bin}" -m pytest -q --confcutdir=tests/config/lwd \
  tests/config/lwd/test_pd_topology.py
cd "${ascend_dir}"
"${python_bin}" -m pytest -q --confcutdir=tests/ut/distributed/lwd_data_plane \
  tests/ut/distributed/lwd_data_plane/test_lwd_host_topology.py
"${python_bin}" -m pytest -q --confcutdir=tests/ut/worker \
  tests/ut/worker/test_lwd_pd_runtime.py \
  tests/ut/worker/test_lwd_v4_mixed_migration.py \
  tests/ut/worker/test_lwd_edge_mm_refill.py \
  tests/ut/worker/test_lwd_s1s2_refactor_guard.py \
  tests/ut/worker/test_lwd_base_weave_guard.py \
  tests/ut/worker/test_lwd_cloud_early_recv.py \
  tests/ut/worker/test_lwd_pre_recv.py
