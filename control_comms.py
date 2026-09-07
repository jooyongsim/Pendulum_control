"""Small newline-delimited JSON interface for PendulumController.ino."""

import json
from enum import Enum
from typing import List, Optional, Tuple

import serial
import serial.tools.list_ports


# The Nucleo re-enumerates on a different COM port fairly often, so the port is
# found by its ST-Link descriptor rather than hard-coded.
STLINK_HINTS = ("stlink", "st-link", "stmicroelectronics")


def find_board_port(preferred: Optional[str] = None) -> str:
    """Return `preferred` if that port exists, otherwise the ST-Link COM port."""
    ports = list(serial.tools.list_ports.comports())
    present = [p.device for p in ports]
    if preferred and preferred in present:
        return preferred

    matches = [p.device for p in ports
               if any(h in f"{p.description} {p.hwid}".lower() for h in STLINK_HINTS)]
    if len(matches) == 1:
        if preferred:
            print(f"{preferred} is not present; using {matches[0]} instead.")
        else:
            print(f"Found the board on {matches[0]}.")
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"No ST-Link serial port found. Ports present: {present or 'none'}. "
            "Check the USB cable, or set the port explicitly."
        )
    raise RuntimeError(f"Several ST-Link ports found ({matches}); set the port explicitly.")


class StatusCode(Enum):
    OK = 0
    ERROR = 1


class DebugLevel(Enum):
    DEBUG_NONE = 0
    DEBUG_ERROR = 1
    DEBUG_WARN = 2
    DEBUG_INFO = 3


class ControlComms:
    def __init__(self, timeout: float = 0.2, debug_level: DebugLevel = DebugLevel.DEBUG_NONE):
        self.ser = serial.Serial(dsrdtr=False)
        self.ser.timeout = timeout
        self.debug_level = debug_level
        self.ser.rts = False
        self.ser.dtr = False

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def get_serial_list(self):
        return tuple((p.device, p.description, p.hwid) for p in serial.tools.list_ports.comports())

    def connect(self, port: str, baud_rate: int = 500000) -> StatusCode:
        self.close()
        self.ser.port = port
        self.ser.baudrate = baud_rate
        try:
            self.ser.open()
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            return StatusCode.OK
        except Exception as exc:
            if self.debug_level.value >= DebugLevel.DEBUG_ERROR.value:
                print(f"Serial open error: {exc}")
            return StatusCode.ERROR

    def step(self, command: int, action: List[float]) -> Optional[Tuple[int, int, bool, Tuple[float, ...]]]:
        if not self.ser.is_open:
            raise RuntimeError("Serial port is not open")

        # A newline makes framing deterministic for ArduinoJson and read_until().
        payload = {"command": int(command), "action": [float(v) for v in action]}
        try:
            self.ser.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
            raw = self.ser.read_until(b"\n")
            if not raw:
                return None
            data = json.loads(raw.decode("utf-8"))
            return (
                int(data["status"]),
                int(data["timestamp"]),
                bool(data["terminated"]),
                tuple(float(v) for v in data["observation"]),
            )
        except Exception as exc:
            if self.debug_level.value >= DebugLevel.DEBUG_ERROR.value:
                print(f"Serial transaction error: {exc}")
            return None
