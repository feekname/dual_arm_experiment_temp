"""Force sensor shared-memory readers for ATI/SOEM and POSIX publishers."""
import ctypes
import errno
import struct
from multiprocessing import shared_memory

import numpy as np


class _SixAxisForceData(ctypes.Structure):
    """Matches ATI SOEM test/linux/6_FT/include/FT_api.h."""
    _fields_ = [
        ("fx", ctypes.c_float), ("fy", ctypes.c_float),
        ("fz", ctypes.c_float), ("tx", ctypes.c_float),
        ("ty", ctypes.c_float), ("tz", ctypes.c_float),
        ("state_code", ctypes.c_int32),
    ]


class ForceSensorReader:
    """Read ATI System V data or a legacy POSIX double array."""

    SHM_RDONLY = 0o10000

    def __init__(self, shm_name="6_axis_force_shm", data_size=6,
                 backend="sysv", shm_key=0x1234):
        if data_size <= 0:
            raise ValueError("data_size 必须为正数")
        if backend not in ("sysv", "posix"):
            raise ValueError("backend 必须是 'sysv' 或 'posix'")
        self.shm_name = shm_name
        self.data_size = data_size
        self.backend = backend
        self.shm_key = shm_key
        self.shm = None
        self.shm_id = -1
        self.shm_addr = None
        self.connected = False
        self.state_code = 0
        self._libc = None

    def connect(self):
        return self._connect_sysv() if self.backend == "sysv" else self._connect_posix()

    def _connect_sysv(self):
        size = ctypes.sizeof(_SixAxisForceData)
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        self._libc.shmget.restype = ctypes.c_int
        self._libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self._libc.shmat.restype = ctypes.c_void_p
        self._libc.shmdt.argtypes = [ctypes.c_void_p]
        self._libc.shmdt.restype = ctypes.c_int

        # flags=0 means the reader never creates an empty/stale segment.
        self.shm_id = self._libc.shmget(self.shm_key, size, 0)
        if self.shm_id == -1:
            err = ctypes.get_errno()
            if err == errno.ENOENT:
                print(f"[SHM] 找不到ATI System V段 key=0x{self.shm_key:x}，请先启动6_FT。")
            else:
                print(f"[SHM] shmget失败: errno={err} ({errno.errorcode.get(err, 'UNKNOWN')})")
            return False

        # ATI通常以root运行；读取端只申请读权限，避免因umask导致shmat(EACCES)。
        self.shm_addr = self._libc.shmat(self.shm_id, None, self.SHM_RDONLY)
        if self.shm_addr == ctypes.c_void_p(-1).value:
            err = ctypes.get_errno()
            self.shm_addr = None
            print(f"[SHM] shmat失败: errno={err} ({errno.errorcode.get(err, 'UNKNOWN')})")
            return False

        self.connected = True
        print(f"[SHM] 已连接ATI System V共享内存: key=0x{self.shm_key:x}, "
              f"ID={self.shm_id}, struct={size} bytes")
        return True

    def _connect_posix(self):
        expected_bytes = 8 * self.data_size
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            print(f"[SHM] POSIX共享内存连接失败 {self.shm_name}: {exc}")
            return False
        if self.shm.size < expected_bytes:
            actual_bytes = self.shm.size
            self.shm.close()
            self.shm = None
            print(f"[SHM] POSIX容量不足：实际{actual_bytes} bytes，需要{expected_bytes} bytes")
            return False
        self.connected = True
        print(f"[SHM] 已连接POSIX共享内存: {self.shm_name}")
        return True

    def read_force(self):
        if not self.connected:
            raise RuntimeError("力传感器共享内存尚未连接")
        if self.backend == "sysv":
            raw = ctypes.string_at(self.shm_addr, ctypes.sizeof(_SixAxisForceData))
            fx, fy, fz, tx, ty, tz, self.state_code = struct.unpack("=6fi", raw)
            return np.array([fx, fy, fz, tx, ty, tz], dtype=float)

        expected_bytes = 8 * self.data_size
        raw = bytes(self.shm.buf[:expected_bytes])
        return np.array(struct.unpack(f"={self.data_size}d", raw), dtype=float)

    def close(self):
        if self.backend == "sysv" and self.shm_addr is not None and self._libc:
            self._libc.shmdt(ctypes.c_void_p(self.shm_addr))
        elif self.shm is not None:
            self.shm.close()
        self.shm = None
        self.shm_addr = None
        self.shm_id = -1
        self.connected = False
