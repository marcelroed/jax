#!/bin/bash
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Install wheels stored in `JAXCI_OUTPUT_DIR` on the system using the Python
# binary set in JAXCI_PYTHON. Use the absolute path to the `find` utility to
# avoid using the Windows version of `find` on Msys.

# If testing a jax only release, only find and install the jax wheel.
if [[ "$JAXCI_TEST_JAX_ONLY_RELEASE" == "1" ]]; then
  WHEEL_SEARCH_PATTERN="-name "*jax*py3*any.whl""
  # If running CUDA tests, append "[cuda]" to the wheel name to install the
  # CUDA plugins.
  if [[ "${BASH_SOURCE[1]}" =~ "*cuda*" ]]; then
    WHEEL_SEARCH_PATTERN="${WHEEL_SEARCH_PATTERN} -printf "%p\[cuda\]\n""
  fi
else
  WHEEL_SEARCH_PATTERN="-name "*jax*py3*any.whl" -o -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*""
fi

WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \( $WHEEL_SEARCH_PATTERN  \)) )

if [[ -z "${WHEELS[@]}" ]]; then
  echo "ERROR: No wheels found under $JAXCI_OUTPUT_DIR"
  exit 1
fi

echo "Installing the following wheels:"
echo "${WHEELS[@]}"

# Install `uv` if it's not already installed. `uv` is much faster than pip for
# installing Python packages.
if ! command -v uv >/dev/null 2>&1; then
  pip install uv~=0.5.30
fi

# On Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  "$JAXCI_PYTHON" -m uv pip install $(cygpath -w "${WHEELS[@]}")
else
  "$JAXCI_PYTHON" -m uv pip install "${WHEELS[@]}"
fi

if [[ "$JAXCI_INSTALL_JAX_CURRENT_COMMIT" == "1" ]]; then
  echo "Installing the JAX package at the current commit..."
  # Install JAX package at the current commit.
  "$JAXCI_PYTHON" -m uv pip install .
fi