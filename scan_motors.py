"""Scan for any Feetech motors on the bus at different baudrates."""
import sys
from scservo_sdk import PortHandler, PacketHandler

PORT = "/dev/tty.usbmodem5A7C1208251"
BAUDRATES = [1_000_000, 500_000, 115_200, 57_600, 38_400, 19_200, 9_600]
PROTOCOL_END = 0  # Feetech uses protocol 0

ph = PortHandler(PORT)
pkt = PacketHandler(PROTOCOL_END)

if not ph.openPort():
    print(f"ERROR: Cannot open port {PORT}")
    sys.exit(1)

print(f"Port {PORT} opened successfully.\n")

for baud in BAUDRATES:
    ph.setBaudRate(baud)
    print(f"--- Scanning at baudrate {baud} ---")
    found = []
    for motor_id in range(0, 20):  # scan IDs 0-19
        model_number, comm_result, error = pkt.ping(ph, motor_id)
        if comm_result == 0:  # COMM_SUCCESS
            found.append((motor_id, model_number))
            print(f"  FOUND motor ID={motor_id}, model={model_number}")
    if not found:
        print("  No motors found at this baudrate.")
    print()

ph.closePort()
print("Scan complete.")
