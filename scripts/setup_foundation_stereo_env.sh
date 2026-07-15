#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fs_root="${repo_root}/third_party/FoundationStereo"
environment_name="${1:-foundation_stereo}"

if [[ ! -f "${fs_root}/environment.yml" ]]; then
  printf 'FoundationStereo submodule is missing at %s\n' "${fs_root}" >&2
  printf 'Run: git submodule update --init --recursive\n' >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required to create the FoundationStereo environment.\n' >&2
  exit 1
fi

conda env create --yes --name "${environment_name}" --file "${fs_root}/environment.yml"
printf 'Created %s. Set FOUNDATION_STEREO_CHECKPOINT to the downloaded checkpoint path.\n' "${environment_name}"
