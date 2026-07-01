"""Quick scan of the leader arm port."""
from scservo_sdk import PortHandler, PacketHandler
PORT = "/dev/tty.usbmodem5AAF2883381"
ph = PortHandler(PORT)
pkt = PacketHandler(0)
ph.openPort()
ph.setBaudRate(1_000_000)
print(f"Scanning {PORT} at 1Mbps...")
for mid in range(0, 10):
    model, result, _ = pkt.ping(ph, mid)
    if result == 0:
        print(f"  FOUND motor ID={mid}, model={model}")
ph.closePort()
print("Done.")
