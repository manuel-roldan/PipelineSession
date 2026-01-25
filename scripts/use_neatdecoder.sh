#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="${REPO_ROOT}/third_party/gst-plugins"
LIB_NAME="libgstneatdecoder.so"

if [[ ! -d "${PLUGIN_DIR}" ]]; then
  echo "Missing ${PLUGIN_DIR}. Run scripts/sync_neatdecoder.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "${PLUGIN_DIR}/${LIB_NAME}" ]]; then
  echo "Missing ${PLUGIN_DIR}/${LIB_NAME}. Run scripts/sync_neatdecoder.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export GST_PLUGIN_PATH="${PLUGIN_DIR}${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}"
if [[ -n "${SIMA_SET_GST_SYSTEM_PATH:-}" ]]; then
  export GST_PLUGIN_SYSTEM_PATH_1_0="${PLUGIN_DIR}${GST_PLUGIN_SYSTEM_PATH_1_0:+:${GST_PLUGIN_SYSTEM_PATH_1_0}}"
fi

echo "GST_PLUGIN_PATH=${GST_PLUGIN_PATH}"
if [[ -n "${SIMA_SET_GST_SYSTEM_PATH:-}" ]]; then
  echo "GST_PLUGIN_SYSTEM_PATH_1_0=${GST_PLUGIN_SYSTEM_PATH_1_0}"
fi
