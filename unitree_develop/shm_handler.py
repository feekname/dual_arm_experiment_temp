# shm_handler.py
import numpy as np
from multiprocessing import shared_memory
import struct

class ForceSensorReader:
    """
    共享内存读取器：对接 1000Hz C++ 传感器数据
    """
    def __init__(self, shm_name: str, data_size: int = 12):
        self.shm_name = shm_name
        self.data_size = data_size # 12个double (8字节/个)
        self.shm = None
        self.connected = False

    def connect(self):
        try:
            # 挂载 C++ 已经创建好的共享内存
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.connected = True
            print(f"[SHM] 成功连接到共享内存: {self.shm_name}")
        except FileNotFoundError:
            print(f"[SHM] 错误：找不到名为 {self.shm_name} 的共享内存，请确保 C++ 程序已启动。")

    def read_force(self):
        if not self.connected:
            return np.zeros(self.data_size)
        
        # 直接从内存中读取数据并解析为 double 数组
        # 'd' 代表 double, 12d 代表 12 个 double
        raw_data = self.shm.buf[:8 * self.data_size]
        data = struct.unpack('12d', raw_data)
        return np.array(data)

    def close(self):
        if self.shm:
            self.shm.close()