#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
release_root="$(cd -- "$script_dir/../.." && pwd -P)"
exec "$release_root/venv/bin/python" "$script_dir/gooaye-telegram-monitor.py"
