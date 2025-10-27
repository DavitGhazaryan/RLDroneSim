#!/usr/bin/env bash
set -euo pipefail

# Fail with a clear message
trap 'echo "ERROR: Line $LINENO failed. Aborting." >&2' ERR

# ---- Helpers ----
log() { echo -e "\n[+] $*\n"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- Preconditions ----
# Ensure we're in the repo root where ./waf lives.
if [[ ! -x "./waf" ]]; then
  echo "ERROR: ./waf not found in current directory. cd to the repo root first." >&2
  exit 1
fi

# ---- Python package cleanup / protobuf correction ----
# Uninstall mavsdk and protobuf from pip (ignore if not present).
if have pip; then
  log "Uninstalling mavsdk (pip) if present..."
  pip uninstall -y mavsdk || true
  log "Uninstalling protobuf (pip) if present..."
  pip uninstall -y protobuf || true
fi

if have pip3; then
  log "Uninstalling mavsdk (pip3) if present..."
  pip3 uninstall -y mavsdk || true
  log "Uninstalling protobuf (pip3) if present..."
  pip3 uninstall -y protobuf || true
fi

# ---- System protobuf install ----
log "Updating apt and installing python3-protobuf (system package)..."
sudo apt-get update -y
sudo apt-get install -y python3-protobuf

# NOTE: Mixing system Python packages and pip can cause version conflicts.
# You asked for python3-protobuf from apt; we stick to that.

# ---- Build SITL target via waf ----
log "Cleaning waf build..."
./waf clean

log "Configuring waf for --board sitl..."
./waf configure --board sitl

log "Building copter target..."
./waf copter

# ---- Build new C++ code under clock_node ----
log "Building clock_node via CMake..."
cd ../clock_node 

mkdir -p build
cd build 

# Use all cores if nproc is available
JOBS=1
if have nproc; then
  JOBS="$(nproc)"
fi

log "Configuring (CMAKE_BUILD_TYPE=Release)..."
cmake -DCMAKE_BUILD_TYPE=Release ..

log "Compiling with make -j${JOBS}..."
make -j "${JOBS}"

# Add Gazebo Transport environment variables to ~/.bashrc if not already present
BASHRC="$HOME/.bashrc"

# Function to add or update variable
add_or_update_var() {
    VAR_NAME=$1
    VAR_VALUE=$2
    if grep -q "^export $VAR_NAME=" "$BASHRC"; then
        sed -i "s|^export $VAR_NAME=.*|export $VAR_NAME=$VAR_VALUE|" "$BASHRC"
    else
        echo "export $VAR_NAME=$VAR_VALUE" >> "$BASHRC"
    fi
}

add_or_update_var GZ_IP 127.0.0.1
add_or_update_var GZ_PARTITION my_partition
add_or_update_var GZ_TRANSPORT_TOPIC_STATISTICS 0
add_or_update_var GZ_VERBOSE 0

echo "Gazebo environment variables added to ~/.bashrc"

log "All done ✔"
