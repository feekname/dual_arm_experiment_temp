import time
from shm_handler import ForceSensorReader

reader = ForceSensorReader(
    backend="sysv",
    shm_key=0x1234,
    data_size=6,
)

if not reader.connect():
    raise SystemExit("共享内存连接失败")

try:
    for _ in range(100):
        force = reader.read_force()
        print(
            "Fx={:+8.3f}, Fy={:+8.3f}, Fz={:+8.3f}, "
            "Tx={:+8.3f}, Ty={:+8.3f}, Tz={:+8.3f}, state={}".format(
                *force, reader.state_code
            )
        )
        time.sleep(0.1)
finally:
    reader.close()