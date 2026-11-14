#!/usr/bin/env bash
# One-time system setup for the inspection cell laptop. Needs root:
#   sudo bash scripts/setup_system.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Run with sudo"; exit 1; }
TARGET_USER="${SUDO_USER:-intel}"

# 1. Native prerequisites from Physical AI Studio's installation guide
#    (ffmpeg is required by torchcodec for dataset recording/decoding),
#    plus Intel GPU (OpenCL + Level Zero) userspace drivers for OpenVINO/torch XPU.
apt-get update
apt-get install -y \
  ffmpeg libgl1 libglib2.0-0t64 libusb-1.0-0 libusb-1.0-0-dev libclang-dev \
  pkg-config build-essential g++ git \
  intel-opencl-icd libze1 libze-intel-gpu1

# 2. Serial access to the SO-101 arms (QinHeng CH343, 1a86:55d3) and RealSense USB access.
usermod -aG dialout,video,render,plugdev "$TARGET_USER"   # effective after re-login
# The file must sort before 73-seat-late.rules, which is where the uaccess tag is
# turned into an ACL; a 99- file sets the tag too late and replugged devices lose access.
rm -f /etc/udev/rules.d/99-so101-realsense.rules
cat > /etc/udev/rules.d/70-so101-realsense.rules <<'RULES'
# SO-101 leader/follower servo bus adapters
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0660", GROUP="dialout", TAG+="uaccess"
# Intel RealSense cameras
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", MODE="0660", GROUP="plugdev", TAG+="uaccess"
RULES
udevadm control --reload-rules
udevadm trigger --action=add --subsystem-match=tty --subsystem-match=usb
udevadm settle
# Belt and braces: grant access right now too.
setfacl -m "u:${TARGET_USER}:rw" /dev/ttyACM* 2>/dev/null || true

echo "Done. Serial ports:"; ls -l /dev/ttyACM*
