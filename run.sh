#!/usr/bin/env bash
# FreshCheck one-command launcher (Linux / macOS / WSL).
#   ./run.sh          -> C++ server (builds it the first time), falls back to Python if it can't build
#   ./run.sh python   -> Python safety-net server
# Then open http://localhost:7860 (opened automatically if possible).
set -e
cd "$(dirname "$0")"
PORT="${PORT:-7860}"
PY="$(command -v python3 || command -v python)"

"$PY" find_models.py || exit 1

open_browser() {  # opens the page once the server answers
  ( for _ in $(seq 1 60); do
      if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        (xdg-open "http://localhost:$PORT" || open "http://localhost:$PORT" || cmd.exe /c start "http://localhost:$PORT") >/dev/null 2>&1
        break
      fi; sleep 1
    done ) &
}

run_python() {
  echo ">> Starting the Python server"
  if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
  ./.venv/bin/pip install -q -r app_fallback/requirements.txt
  open_browser
  exec ./.venv/bin/python app_fallback/server.py --port "$PORT"
}

build_cpp() {
  [ -x app/build/freshcheck ] && return 0
  command -v cmake >/dev/null && command -v c++ >/dev/null || { echo "!! cmake / C++ compiler not found"; return 1; }
  ORT_VER=1.20.1
  ORT_DIR="${ORT_DIR:-$PWD/.onnxruntime}"
  if [ ! -d "$ORT_DIR/lib" ]; then
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64)  PKG=onnxruntime-linux-x64-$ORT_VER ;;
      Linux-aarch64) PKG=onnxruntime-linux-aarch64-$ORT_VER ;;
      Darwin-arm64)  PKG=onnxruntime-osx-arm64-$ORT_VER ;;
      Darwin-x86_64) PKG=onnxruntime-osx-x86_64-$ORT_VER ;;
      *) echo "!! unsupported platform"; return 1 ;;
    esac
    echo ">> Downloading ONNX Runtime $ORT_VER (one time)"
    mkdir -p "$ORT_DIR"
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/$PKG.tgz" \
      | tar -xz -C "$ORT_DIR" --strip-components=1 || return 1
  fi
  echo ">> Building the C++ server (one time, ~1 min)"
  cmake -S app -B app/build -DONNXRUNTIME_DIR="$ORT_DIR" >/dev/null || return 1
  cmake --build app/build -j >/dev/null || return 1
}

if [ "$1" = "python" ]; then run_python; fi

if build_cpp; then
  echo ">> Starting the C++ server on http://localhost:$PORT  (Ctrl+C to stop)"
  open_browser
  exec ./app/build/freshcheck --config app/config.json --port "$PORT"
else
  echo "!! C++ build not possible here (need: cmake, a C++ compiler, OpenCV: sudo apt install build-essential cmake libopencv-dev  |  brew install cmake opencv)"
  echo "!! Using the Python server instead - same app, same UI."
  run_python
fi
