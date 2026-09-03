# driver.py
import threading
from typing import Optional
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from config import RobotState

class G1Driver:
    """
    G1 硬件驱动类（单例）
    负责底层的 DDS 通讯和数据解析
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(G1Driver, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'): return
        self._initialized = True
        
        # 内部状态
        self.state = RobotState()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc_calculator = CRC()
        
        # 通讯句柄
        self.publisher: Optional[ChannelPublisher] = None
        self.subscriber: Optional[ChannelSubscriber] = None
        
        # 标志位
        self.has_received_state = False
        self._lock = threading.Lock()

        # 存储当前的模式状态机，这是指令被机器人接受的前提
        self.current_mode_machine = 0

    def init_comm(self, network_interface: str):
        """初始化 DDS 通道"""

        try:
            ChannelFactoryInitialize(0, network_interface)
            print(f"[Driver] ChannelFactory 已在接口 {network_interface} 上初始化")
        except Exception as e:
            print(f"[Driver] ChannelFactory 初始化失败： {e}")
            return

        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        # 使用回调函数实时更新状态
        self.subscriber.Init(self._low_state_handler, 10)

    def _low_state_handler(self, msg: LowState_):
        """低层状态回调，将数据解包到 RobotState 对象"""
        with self._lock:
            # for i in range(len(msg.motor_state)-1):
            self.current_mode_machine = msg.mode_machine

            for i in range(29):
                self.state.q[i] = msg.motor_state[i].q
                self.state.dq[i] = msg.motor_state[i].dq
                self.state.ddq[i] = msg.motor_state[i].ddq
                self.state.tau[i] = msg.motor_state[i].tau_est
            
            if not self.has_received_state:
                self.has_received_state = True

    def publish_cmd(self):
        """计算 CRC 并发布指令"""
        self.low_cmd.crc = self.crc_calculator.Crc(self.low_cmd)
        if self.publisher:
            self.low_cmd.mode_machine = self.current_mode_machine
            self.publisher.Write(self.low_cmd)

        # print(self.low_cmd)