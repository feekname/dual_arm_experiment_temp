# config.py
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

# 关节序号
class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

@dataclass
class ArmConfig:
    """机械臂控制配置"""
    # 宇树默认 PD 控制参数
    kp: float = 60.0
    kd: float = 1.5
    # 关节索引组
    left_arm_idx: List[int] = field(default_factory=lambda: [15, 16, 17, 18, 19, 20, 21])
    right_arm_idx: List[int] = field(default_factory=lambda: [22, 23, 24, 25, 26, 27, 28])
    waist_idx: List[int] = field(default_factory=lambda: [12, 13, 14])
    
    # 协同控制频率 (200Hz)
    control_dt: float = 0.005

@dataclass
class RobotState:
    """实时存储机器人的当前状态快照"""
    q: np.ndarray = field(default_factory=lambda: np.zeros(30))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(30))
    ddq: np.ndarray = field(default_factory=lambda: np.zeros(30))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(30))