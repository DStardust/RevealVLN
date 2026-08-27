#!/usr/bin/env bash

workspace=/mnt/daiyang/vla
runtime="$workspace/.remote_runtime"
if [[ ! -x "$runtime/etpr1/bin/python" || ! -d "$runtime/habitat-sim" ]]; then
  echo "remote VLA runtime is not ready" >&2
  return 1 2>/dev/null || exit 1
fi

export VLA_PROJECT_ROOT="$workspace"
export VLA_HABITAT_SIM_ROOT="$runtime/habitat-sim"
export VIRTUAL_ENV="$runtime/etpr1"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$workspace:$runtime/habitat-sim:$workspace/third_party/ETP-R1${PYTHONPATH:+:$PYTHONPATH}"
cd "$workspace"
echo "RevealNav remote runtime active: $VIRTUAL_ENV"
