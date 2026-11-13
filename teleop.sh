#!/bin/sh
cd "$(dirname "$0")"
exec uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B910431681 --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=/dev/cu.usbmodem5B610369821 --teleop.id=leader \
  "$@"
