"""Scan ALL available serial ports for leader arm motors at 1Mbps.
Auto-detects COM ports on Windows (no hardcoded paths).
"""
import sys

from serial.tools import list_ports
from scservo_sdk import PortHandler, PacketHandler


def main():
    ports = list(list_ports.comports())
    if not ports:
        print("ERROR: No serial ports found! Is the USB cable plugged in?")
        sys.exit(1)

    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device} — {p.description} [HWID: {p.hwid}]")

    pkt = PacketHandler(0)

    for p in ports:
        ph = PortHandler(p.device)
        if not ph.openPort():
            print(f"\n  Cannot open {p.device}, skipping.")
            continue

        ph.setBaudRate(1_000_000)
        print(f"\nScanning {p.device} at 1Mbps...")
        found = []
        for mid in range(0, 10):
            model, result, _ = pkt.ping(ph, mid)
            if result == 0:
                found.append((mid, model))
                print(f"  FOUND motor ID={mid}, model={model}")

        if not found:
            print(f"  No motors found on {p.device}")

        ph.closePort()

    print("\nDone.")


if __name__ == "__main__":
    main()
