"""Scan USB serial ports (skip Bluetooth) for Feetech motors at different baudrates."""
import sys

from serial.tools import list_ports
from scservo_sdk import PortHandler, PacketHandler

BAUDRATES = [1_000_000, 500_000, 115_200, 57_600, 38_400, 19_200, 9_600]
PROTOCOL_END = 0  # Feetech uses protocol 0


def scan_port(port_name: str) -> dict:
    """Scan a single port for motors. Returns {baud: [(id, model), ...]}."""
    try:
        ph = PortHandler(port_name)
        pkt = PacketHandler(PROTOCOL_END)

        if not ph.openPort():
            print(f"  ERROR: Cannot open port {port_name}")
            return {}

        print(f"\n{'='*60}")
        print(f"  Scanning port: {port_name}")
        print(f"{'='*60}")

        results = {}
        for baud in BAUDRATES:
            ph.setBaudRate(baud)
            found = []
            for motor_id in range(0, 20):  # scan IDs 0-19
                model_number, comm_result, error = pkt.ping(ph, motor_id)
                if comm_result == 0:  # COMM_SUCCESS
                    found.append((motor_id, model_number))
                    print(f"  [BAUD {baud}] FOUND motor ID={motor_id}, model={model_number}")
            if found:
                results[baud] = found

        ph.closePort()
        return results
    except Exception as e:
        print(f"  ERROR scanning {port_name}: {e}")
        return {}


def main():
    # List all available serial ports
    ports = list(list_ports.comports())
    if not ports:
        print("ERROR: No serial ports found! Is the USB cable plugged in?")
        sys.exit(1)

    print("All serial ports:")
    usb_ports = []
    for p in ports:
        is_usb = "USB" in p.hwid.upper() or "CH34" in p.description.upper()
        marker = " ** USB **" if is_usb else " (Bluetooth/other - skipping)"
        print(f"  {p.device} — {p.description} {marker}")
        if is_usb:
            usb_ports.append(p)

    if not usb_ports:
        print("\nERROR: No USB serial ports found! Only Bluetooth ports detected.")
        print("Make sure the robot arm USB cables are plugged in.")
        sys.exit(1)

    print(f"\nFound {len(usb_ports)} USB port(s). Scanning for motors...")

    # Scan only USB ports
    all_results = {}
    for p in usb_ports:
        results = scan_port(p.device)
        if results:
            all_results[p.device] = results

    # Summary
    print(f"\n{'='*60}")
    print("  SCAN SUMMARY")
    print(f"{'='*60}")
    if not all_results:
        print("  No motors found on any USB port!")
    else:
        for port, baud_results in all_results.items():
            for baud, motors in baud_results.items():
                print(f"  Port {port} @ {baud} baud: {len(motors)} motors")
                for mid, model in motors:
                    print(f"    Motor ID={mid}, Model={model}")
    print("\nScan complete.")


if __name__ == "__main__":
    main()
