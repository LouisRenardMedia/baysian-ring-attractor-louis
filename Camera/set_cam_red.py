import os
import subprocess

DEVICE = "/dev/video0"

# All the v4l2 controls to apply
commands = [
    f"v4l2-ctl -d {DEVICE} --set-ctrl=white_balance_auto_preset=1",
    # f"v4l2-ctl -d {DEVICE} --set-ctrl=red_balance=1500",
    # f"v4l2-ctl -d {DEVICE} --set-ctrl=blue_balance=1000",
]

print("🔧 Applying camera parameters...")

for cmd in commands:
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"⚠️  Failed: {cmd}")

print("✅ Camera parameters set successfully.")