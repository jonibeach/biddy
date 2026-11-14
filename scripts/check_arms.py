"""Read-only check: ping the 6 Feetech servos on each SO-101 serial port and
print their positions and bus voltage. Does not enable torque or move anything.

Run with the Studio backend's environment:
    ~/physical-ai-studio/application/backend/.venv/bin/python scripts/check_arms.py
"""
import glob
import sys

import scservo_sdk as scs

ok = True
for port in sorted(glob.glob("/dev/ttyACM*")):
    ph = scs.PortHandler(port)
    try:
        ph.openPort()
        ph.setBaudRate(1_000_000)
    except Exception as e:  # noqa: BLE001
        print(f"{port}: CANNOT OPEN ({e})")
        ok = False
        continue
    pk = scs.PacketHandler(0)
    found = []
    for mid in range(1, 7):
        _, res, _ = pk.ping(ph, mid)
        if res == scs.COMM_SUCCESS:
            pos, _, _ = pk.read2ByteTxRx(ph, mid, 56)   # Present_Position
            volt, _, _ = pk.read1ByteTxRx(ph, mid, 62)  # Present_Voltage (0.1 V)
            found.append((mid, pos, volt / 10))
    ph.closePort()
    print(f"{port}: {len(found)}/6 servos", " ".join(f"id{m}=pos{p}/{v}V" for m, p, v in found))
    ok &= len(found) == 6
sys.exit(0 if ok else 1)
