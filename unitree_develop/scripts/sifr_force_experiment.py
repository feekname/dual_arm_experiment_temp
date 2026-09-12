'''
Author: pengfei 524560850@qq.com
Date: 2026-09-08
Description: 双臂协作搬运实验 — SIFR (力传感器版本, 对比实验)
             Static Internal Force Regulation: 期望内力固定为F_fixed，不动态调节
             与DIFR版本对比，验证公式41动态调节内力的优势

与DIFR版本的唯一区别:
  - 期望内力 F_I_des 始终等于 F_fixed (标称值)，不随外力/滑动状态变化
  - 仍然通过肩关节roll跟踪 F_fixed
  - 滑动位移δ仍然估计(用于记录和对比)，但不用于调节内力

仅使用左手六维力传感器；共享内存可以是6维（仅左手）或兼容旧版12维布局。

实验三阶段:
  阶段1 MOVE  (0~10s): 双臂五次多项式运动到搬运预备姿态
  阶段2 GRASP (10~20s): 保持姿态，GMO稳态偏置采集
  阶段3 TEST  (20~50s): 碰撞测试，期望内力保持固定

用法:
  python sifr_force_experiment.py \
      --left_q  -0.8 0.7 -0.7 0.4 0.0 -0.6 0.0 \
      --right_q  0.8 0.7  0.7 0.4 0.0  0.6 0.0 \
      --shm-name 6_axis_force_shm \
      --F-fixed 16.35 --mu 0.3 --m-object 1.0
'''
import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import pybullet as p
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server
from shm_handler import ForceSensorReader
from normal_ik_actuator import SymmetricNormalIKActuator


# ================================================================
# 力传感器数据容器
# ================================================================
class ForceData:
    AXIS = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, normal_axis='z', tangent_axis='y',
                 normal_sign=1.0, tangent_sign=1.0):
        self.left = np.zeros(6)
        self.left_bias = np.zeros(6)
        self.bias_applied = False
        self.normal_idx = self.AXIS[normal_axis]
        self.tangent_idx = self.AXIS[tangent_axis]
        self.normal_sign = normal_sign
        self.tangent_sign = tangent_sign

    def update(self, raw_data):
        raw_data = np.asarray(raw_data, dtype=float).reshape(-1)
        if raw_data.size < 6:
            raise ValueError(f"左手六维力数据长度不足: {raw_data.size}")
        self.left = raw_data[:6].copy()

    def get_forces(self):
        l = self.left - self.left_bias if self.bias_applied else self.left
        return l

    def get_internal_force(self):
        '''单侧接触的法向力即每只手的内力幅值。'''
        l = self.get_forces()
        signed_normal = self.normal_sign * l[self.normal_idx]
        return max(0.0, signed_normal), signed_normal

    def get_external_force_from_sensor(self):
        '''返回选定切向轴上的有符号接触力。'''
        l = self.get_forces()
        return self.tangent_sign * l[self.tangent_idx]


# ================================================================
# GMO 观测器
# ================================================================
class GMOObserver:
    def __init__(self, n_dof=7, gain=5.0):
        self.n_dof = n_dof
        self.K = gain * np.eye(n_dof)
        self.P_hat = np.zeros(n_dof)
        self.r = np.zeros(n_dof)
        self.M_last = None
        self._initialized = False

    def reset(self):
        self.P_hat = np.zeros(self.n_dof)
        self.r = np.zeros(self.n_dof)
        self.M_last = None
        self._initialized = False

    def update(self, M, Cq, G, tau, dq, dt):
        P = M @ dq
        if self._initialized and self.M_last is not None:
            dM = (M - self.M_last) / max(dt, 1e-6)
        else:
            dM = np.zeros_like(M)
        CTq = dM @ dq - Cq
        self.M_last = M.copy()
        self._initialized = True
        P_hat_dot = CTq + self.r + tau - G
        self.P_hat += P_hat_dot * dt
        self.r = self.K @ (P - self.P_hat)
        return self.r.copy()


# ================================================================
# SIFR 控制器（固定期望内力，对比用）
# ================================================================
class SIFRController:
    '''
    Static Internal Force Regulation: 期望内力固定为F_fixed
    仍然估计滑动位移δ（用于记录对比），但不用于调节内力
    '''
    def __init__(self, F_fixed=16.35, mu=0.3, m_object=1.0, K_f=0.002,
                 K_p=0.0, roll_offset_max=0.10, force_deadband=0.2,
                 roll_offset_min=None, delta_min=0.0, delta_max=0.05,
                 delta_dot_max=0.5):
        self.F_fixed = F_fixed
        self.mu = mu
        self.m_object = m_object
        self.K_f = K_f
        self.K_p = K_p
        self.roll_offset_max = roll_offset_max
        self.roll_offset_min = (-roll_offset_max if roll_offset_min is None
                                else roll_offset_min)
        self.force_deadband = force_deadband
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_dot_max = delta_dot_max
        self.delta = 0.0
        self.delta_dot = 0.0
        self.roll_offset = 0.0

    def reset(self):
        self.delta = 0.0
        self.delta_dot = 0.0
        self.roll_offset = 0.0

    def update(self, F_E, F_I_est, dt):
        '''
        SIFR更新：期望内力固定，仅估计滑动状态用于记录
        Returns:
            F_I_des(=F_fixed), delta_roll, delta, fc, active(始终False)
        '''
        # 期望内力固定
        F_I_des = self.F_fixed

        # 估计滑动位移（开环，用于记录对比）
        force_demand = abs(F_E)
        # 状态估计必须使用传感器实际实现的内力，而不是控制器期望值。
        # 否则在真机跟踪滞后时会错误假设摩擦力已经建立。
        available_friction = self.mu * max(float(F_I_est), 0.0)
        net_force = force_demand - available_friction
        if self.delta_dot <= 0.0 and net_force <= 0.0:
            self.delta_dot = 0.0  # 静摩擦区：不允许“反向滑移”伪影
        else:
            self.delta_dot = np.clip(
                self.delta_dot + net_force / self.m_object * dt,
                0.0, self.delta_dot_max)
        self.delta = np.clip(
            self.delta + self.delta_dot * dt, self.delta_min, self.delta_max)
        # 摩擦锥裕度
        fc = force_demand / max(self.mu * F_I_des, 1e-6)

        # PI闭合控制。比例项改善碰撞后的即时响应；最终发送仍经过
        # 单帧关节变化限幅，因此不会把比例阶跃直接发送到真机。
        force_error = F_I_des - F_I_est
        if abs(force_error) < self.force_deadband:
            force_error = 0.0
        integral_candidate = self.roll_offset + self.K_f * force_error * dt
        command_unclipped = integral_candidate + self.K_p * force_error
        delta_roll = float(np.clip(
            command_unclipped, self.roll_offset_min, self.roll_offset_max))
        # 条件积分抗饱和：误差继续把指令推向饱和端时暂停积分。
        driving_further = ((command_unclipped > self.roll_offset_max and force_error > 0.0) or
                           (command_unclipped < self.roll_offset_min and force_error < 0.0))
        if not driving_further:
            self.roll_offset = float(np.clip(
                integral_candidate, self.roll_offset_min, self.roll_offset_max))

        return F_I_des, delta_roll, self.delta, fc, False


# ================================================================
# 双臂 PyBullet 模型
# ================================================================
class DualArmPyBullet:
    def __init__(self, urdf_path):
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(os.getcwd(), urdf_path)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.all_revolute = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.all_revolute.append(info[0])
        self.n_total = len(self.all_revolute)
        self.joint_to_pos = {jid: idx for idx, jid in enumerate(self.all_revolute)}
        self.left_joints = []
        self.right_joints = []
        for jid in self.all_revolute:
            name = p.getJointInfo(self.robot_id, jid)[1].decode().lower()
            if name.startswith("left_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.left_joints.append(jid)
            elif name.startswith("right_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.right_joints.append(jid)
        if len(self.left_joints) != 7:
            self.left_joints = self.all_revolute[:7]
        if len(self.right_joints) != 7:
            self.right_joints = self.all_revolute[-7:]
        link_to_id = {
            p.getJointInfo(self.robot_id, jid)[12].decode(): jid
            for jid in range(p.getNumJoints(self.robot_id))
        }
        try:
            self.left_ee = link_to_id["left_base_link"]
            self.right_ee = link_to_id["right_base_link"]
        except KeyError as exc:
            raise RuntimeError(
                "URDF中未找到IK末端left_base_link/right_base_link") from exc
        self.left_pos = [self.joint_to_pos[jid] for jid in self.left_joints]
        self.right_pos = [self.joint_to_pos[jid] for jid in self.right_joints]
        print(f"[PyBullet] 左臂: {self.left_joints}, 右臂: {self.right_joints}")
        print(f"[PyBullet] IK末端: left_base_link={self.left_ee}, "
              f"right_base_link={self.right_ee}")

    def _build_full(self, arm, q7, dq7=None):
        full_q = np.zeros(self.n_total)
        pos = self.left_pos if arm == "left" else self.right_pos
        for i, p_idx in enumerate(pos):
            full_q[p_idx] = q7[i]
        if dq7 is not None:
            full_dq = np.zeros(self.n_total)
            for i, p_idx in enumerate(pos):
                full_dq[p_idx] = dq7[i]
            return full_q, full_dq
        return full_q

    def update_states(self, left_q, right_q):
        for i, jidx in enumerate(self.left_joints):
            p.resetJointState(self.robot_id, jidx, left_q[i])
        for i, jidx in enumerate(self.right_joints):
            p.resetJointState(self.robot_id, jidx, right_q[i])

    def compute_dynamics(self, arm, q, dq):
        full_q, full_dq = self._build_full(arm, q, dq)
        zero = [0.0] * self.n_total
        pos = self.left_pos if arm == "left" else self.right_pos
        G_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), zero, zero))
        Cq_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), full_dq.tolist(), zero)) - G_full
        G = G_full[pos]
        Cq = Cq_full[pos]
        M = []
        for k in range(7):
            aa = np.zeros(self.n_total)
            aa[pos[k]] = 1.0
            T_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), zero, aa.tolist())) - G_full
            M.append(T_full[pos])
        return np.array(M).T, Cq, G

    def compute_jacobian(self, arm, q, dq):
        full_q, full_dq = self._build_full(arm, q, dq)
        ee = self.left_ee if arm == "left" else self.right_ee
        pos = self.left_pos if arm == "left" else self.right_pos
        J_p, _ = p.calculateJacobian(self.robot_id, ee, [0,0,0],
                                       full_q.tolist(), full_dq.tolist(), [0.0]*self.n_total)
        return np.array(J_p)[:, pos]

    def disconnect(self):
        p.disconnect(self.client)


# ================================================================
# 工具函数
# ================================================================
def quintic_interpolate(start, goal, t, duration):
    ratio = np.clip(t / duration, 0.0, 1.0)
    s = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
    return start + (goal - start) * s

def check_joint_limits(q, q_min, q_max):
    return np.clip(q, q_min, q_max)

def check_velocity_limits(dq, dq_max=2.0):
    return np.clip(dq, -dq_max, dq_max)


ARM_JOINT_NAMES = (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def joint_error_details(actual, goal):
    error = np.asarray(actual, float) - np.asarray(goal, float)
    index = int(np.argmax(np.abs(error)))
    return (float(np.abs(error[index])), ARM_JOINT_NAMES[index],
            float(error[index]))


def update_position_bias(bias, error, dt, gain, bias_limit, deadband):
    """Integral outer loop that still outputs position commands only."""
    active_error = np.where(np.abs(error) > deadband, error, 0.0)
    return np.clip(
        bias + gain * active_error * float(np.clip(dt, 1e-4, 0.05)),
        -bias_limit, bias_limit)


def joint_delta_limit_for_phase(phase, actuation_mode, joint_delta_max,
                                ik_joint_delta_max, dt,
                                move_joint_speed_max):
    """Return the command slew limit for the current experiment phase.

    The tighter IK limit protects force-driven closure in GRASP/TEST only.
    Applying it during MOVE can prevent the arms from reaching the requested
    preparation pose when the outer loop runs below its nominal frequency.
    """
    if phase == 0:
        # joint_delta_max原本按100 Hz设计；改为速度限幅后，即使动力学
        # 计算使外层循环降频，MOVE的允许速度也不再随循环频率下降。
        return move_joint_speed_max * float(np.clip(dt, 1e-4, 0.05))
    if actuation_mode == 'ik':
        return min(joint_delta_max, ik_joint_delta_max)
    return joint_delta_max


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="SIFR双臂协作搬运实验（力传感器版本，对比实验）")
    parser.add_argument('--left_q', type=float, nargs=7, required=True)
    parser.add_argument('--right_q', type=float, nargs=7, required=True)
    parser.add_argument('--move-time', type=float, default=10.0)
    parser.add_argument('--move-settle-time', type=float, default=0.5,
                        help='MOVE结束后等待真机稳定再锁定实际参考姿态(s)')
    parser.add_argument('--move-joint-speed-max', type=float, default=0.8,
                        help='MOVE阶段单关节最大命令速度(rad/s)')
    parser.add_argument('--move-convergence-timeout', type=float, default=5.0,
                        help='规划结束后继续沿相同控制路径等待到位的超时(s)')
    parser.add_argument('--move-goal-tolerance', type=float, default=0.05,
                        help='进入GRASP前允许的最大单关节到位误差(rad)')
    parser.add_argument('--move-position-ki', type=float, default=0.8,
                        help='MOVE终点纯位置外环积分增益(1/s)，设0可关闭')
    parser.add_argument('--move-position-bias-max', type=float, default=0.08,
                        help='外环积分允许附加到位置命令的最大偏置(rad)')
    parser.add_argument('--move-position-deadband', type=float, default=0.003,
                        help='停止积分的位置误差死区(rad)')
    parser.add_argument('--grasp-time', type=float, default=10.0)
    parser.add_argument('--test-time', type=float, default=30.0)
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--gmo-gain', type=float, default=15.0)
    # 力传感器参数
    parser.add_argument('--shm-name', type=str, default='6_axis_force_shm')
    parser.add_argument('--shm-backend', choices=['sysv', 'posix'], default='sysv',
                        help='ATI/SOEM使用sysv；旧POSIX发布端使用posix')
    parser.add_argument('--shm-key', type=lambda x: int(x, 0), default=0x1234,
                        help='ATI FT_api.h中的System V key，默认0x1234')
    parser.add_argument('--force-bias-samples', type=int, default=200)
    parser.add_argument('--normal-axis', choices='xyz', default='z')
    parser.add_argument('--tangent-axis', choices='xyz', default='x')
    parser.add_argument('--normal-sign', type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument('--tangent-sign', type=float, choices=[-1.0, 1.0], default=-1.0)
    # SIFR参数
    parser.add_argument('--F-fixed', type=float, default=16.35,
                        help='固定期望内力 (N), 默认16.35=1kg*9.81/(2*0.3)')
    parser.add_argument('--mu', type=float, default=0.3)
    parser.add_argument('--m-object', type=float, default=1.0)
    parser.add_argument('--K-f', type=float, default=0.002,
                        help='roll模式的力误差积分增益，单位rad/(N*s)')
    parser.add_argument('--actuation-mode', choices=['roll', 'ik'], default='roll',
                        help='roll=旧肩关节方案；ik=双手沿连线法向对称闭合')
    parser.add_argument('--roll-offset-max', type=float, default=0.10,
                        help='左右肩roll相对偏移总量上限(rad)，每侧使用一半')
    parser.add_argument('--force-deadband', type=float, default=0.2,
                        help='内力误差死区(N)，用于减小稳态抖动')
    parser.add_argument('--roll-action-sign', type=float, choices=[-1.0, 1.0], default=1.0,
                        help='若增大指令反而减小夹持力，设为-1')
    parser.add_argument('--ik-max-displacement', type=float, default=0.015,
                        help='IK模式每只手最大法向位移(m)，两手总闭合量为其2倍')
    parser.add_argument('--ik-force-gain', type=float, default=0.0002,
                        help='IK模式力误差积分增益，单位m/(N*s)')
    parser.add_argument('--force-kp', type=float, default=0.0,
                        help='roll模式力误差比例增益，单位rad/N')
    parser.add_argument('--ik-force-kp', type=float, default=0.0001,
                        help='IK模式力误差比例增益，单位m/N；发送端仍有逐帧限幅')
    parser.add_argument('--delta-min', type=float, default=0.0)
    parser.add_argument('--delta-max', type=float, default=0.05)
    parser.add_argument('--delta-dot-max', type=float, default=0.5,
                        help='滑移状态速度上限(m/s)，仅为估计器防发散')
    parser.add_argument('--ik-damping', type=float, default=0.04)
    parser.add_argument('--ik-iterations', type=int, default=160)
    parser.add_argument('--ik-max-step', type=float, default=0.005,
                        help='启动时IK每次迭代的单关节最大步长(rad)')
    # 关节角增量限幅
    parser.add_argument('--joint-delta-max', type=float, default=0.008)
    parser.add_argument('--ik-joint-delta-max', type=float, default=0.001,
                        help='IK模式每帧单关节最大变化量(rad)，并与joint-delta-max取较小值')
    parser.add_argument('--output', type=str, default='sifr_force.txt')
    parser.add_argument('--urdf', type=str, default='description/g1_14dof_brainco_hand.urdf')
    args = parser.parse_args()
    if args.actuation_mode == 'ik':
        if not (0.0 < args.ik_max_displacement <= 0.03):
            parser.error('--ik-max-displacement必须在(0, 0.03] m内；更大位移需先重新仿真验证')
        if args.ik_force_gain <= 0.0 or args.ik_force_kp < 0.0 or args.ik_joint_delta_max <= 0.0:
            parser.error('--ik-force-gain和--ik-joint-delta-max必须为正数')
    if args.delta_max <= args.delta_min or args.delta_dot_max <= 0.0:
        parser.error('--delta-max必须大于--delta-min，且--delta-dot-max必须为正数')
    if (args.move_settle_time < 0.0 or args.move_joint_speed_max <= 0.0 or
            args.move_convergence_timeout <= 0.0 or
            args.move_goal_tolerance <= 0.0 or args.move_position_ki < 0.0 or
            args.move_position_bias_max <= 0.0 or
            args.move_position_deadband < 0.0):
        parser.error('MOVE稳定时间不能为负，速度、到位超时和容差必须为正数')

    print("=" * 80)
    print("  SIFR 双臂协作搬运实验 (力传感器版本, 对比实验)")
    print("  Static Internal Force Regulation: 期望内力固定")
    print("=" * 80)
    print(f"  阶段1 MOVE:  {args.move_time}s")
    print(f"  阶段2 GRASP: {args.grasp_time}s (GMO去偏采集)")
    print(f"  阶段3 TEST:  {args.test_time}s (期望内力固定={args.F_fixed}N)")
    print(f"  力传感器共享内存: {args.shm_name}")
    print(f"  执行方式: {args.actuation_mode.upper()}")
    print("=" * 80)

    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sifr_force_experiment_{ts}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    q_min = np.array([-2.0, -1.2, -2.5, -0.7, -1.6, -1.6, -1.2])
    q_max = np.array([ 2.0,  1.2,  2.5,  1.8,  1.6,  1.6,  1.2])
    dq_max = 0.2

    # 初始化力传感器
    print(f"\n[力传感器] 连接共享内存: {args.shm_name}")
    force_reader = ForceSensorReader(args.shm_name, data_size=6,
                                     backend=args.shm_backend,
                                     shm_key=args.shm_key)
    if not force_reader.connect():
        raise RuntimeError("左手六维力传感器共享内存未就绪，实验未启动。")
    force_data = ForceData(args.normal_axis, args.tangent_axis,
                           args.normal_sign, args.tangent_sign)

    pb = DualArmPyBullet(args.urdf)
    gmo_left = GMOObserver(7, args.gmo_gain)
    gmo_right = GMOObserver(7, args.gmo_gain)
    action_max = (2.0 * args.ik_max_displacement
                  if args.actuation_mode == 'ik' else args.roll_offset_max)
    force_gain = args.ik_force_gain if args.actuation_mode == 'ik' else args.K_f
    force_kp = args.ik_force_kp if args.actuation_mode == 'ik' else args.force_kp
    sifr = SIFRController(F_fixed=args.F_fixed, mu=args.mu,
                           m_object=args.m_object, K_f=force_gain, K_p=force_kp,
                           roll_offset_max=action_max,
                           force_deadband=args.force_deadband,
                           roll_offset_min=(0.0 if args.actuation_mode == 'ik' else None),
                           delta_min=args.delta_min, delta_max=args.delta_max,
                           delta_dot_max=args.delta_dot_max)

    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)
    if (np.any(goal_l < q_min) or np.any(goal_l > q_max) or
            np.any(goal_r < q_min) or np.any(goal_r > q_max)):
        parser.error('MOVE目标超出实验代码的关节安全限位')
    print("[MOVE目标] 关节顺序: " + " ".join(ARM_JOINT_NAMES))
    print("[MOVE目标] 左臂: " + " ".join(f"{v:+.3f}" for v in goal_l))
    print("[MOVE目标] 右臂: " + " ".join(f"{v:+.3f}" for v in goal_r))
    ik_actuator = None
    # IK必须在MOVE结束后以真机实际到达姿态建立。此处不能使用理论goal，
    # 否则cmd/limit与sent/actual会落在两个不同的零点上。

    print(f"\n[初始化] 连接 G1...")
    server = G1Server(network_interface=args.interface, shm_name=args.shm_name)
    server.start()
    time.sleep(0.5)
    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    # 力传感器零偏采集
    print(f"\n[力传感器] 采集零偏 ({args.force_bias_samples}样本)...")
    bias_samples_l = []
    for i in range(args.force_bias_samples):
        raw = force_reader.read_force()
        bias_samples_l.append(raw[0:6].copy())
        time.sleep(0.01)
    force_data.left_bias = np.mean(bias_samples_l, axis=0)
    force_data.bias_applied = True
    print(f"[力传感器] 左手零偏: {force_data.left_bias}")

    data_file = open(output_path, 'w')
    data_file.write("# SIFR双臂协作搬运实验数据（力传感器版本，对比实验）\n")
    data_file.write(f"# 方法: SIFR (Static Internal Force Regulation, 期望内力固定)\n")
    data_file.write(f"# 力传感器: 共享内存 {args.shm_name}\n")
    data_file.write(f"# 时间: {datetime.now()}\n")
    data_file.write(f"# 阶段: MOVE={args.move_time}s, GRASP={args.grasp_time}s, TEST={args.test_time}s\n")
    data_file.write(f"# 参数: F_fixed={args.F_fixed}, mu={args.mu}, m_object={args.m_object}\n")
    data_file.write(f"# actuation_mode: {args.actuation_mode}\n")
    data_file.write(f"# action_offset_unit: {'m_total_closure' if args.actuation_mode == 'ik' else 'rad_total_roll'}\n")
    data_file.write(f"# ik_max_displacement_per_hand_m: {args.ik_max_displacement}\n")
    data_file.write(f"# force_PI: Ki={force_gain}, Kp={force_kp}; "
                    f"delta_limits=[{args.delta_min},{args.delta_max}], "
                    f"delta_dot_max={args.delta_dot_max}\n")
    data_file.write(
        f"# MOVE纯位置外环: Ki={args.move_position_ki}, "
        f"bias_max={args.move_position_bias_max}rad, "
        f"deadband={args.move_position_deadband}rad, torque_ff=0\n")
    data_file.write("# 列: time phase(0=MOVE,1=GRASP,2=TEST) "
                     "left_q(7) right_q(7) "
                     "left_gmo_F(3) right_gmo_F(3) "
                     "force_left_normal "
                     "F_I_est(内力估计,N) F_E_corrected(TEST切向碰撞力,N) F_E_sensor_raw(N) "
                     "F_I_des(=F_fixed) delta(滑动位移) action_offset(执行偏移) "
                     "fc(摩擦锥裕度) sifr_active(始终0) "
                     "force_left_6dof(6) "
                     "actual_total_closure(m) closure_limit(m) "
                     "left_joint_tracking_rmse(rad) right_joint_tracking_rmse(rad) "
                     "sent_target_total_closure(m)\n")
    data_file.write("#" + "=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("  实验开始")
    print("=" * 80)

    t_step = 0.01
    exp_start = time.time()
    last_print = 0.0
    record_count = 0
    last_time = time.time()
    gmo_left.reset()
    gmo_right.reset()
    sifr.reset()

    # 关节角增量限幅
    last_target_l = start_l.copy()
    last_target_r = start_r.copy()
    # IK实际闭合量以GRASP首帧的真机姿态为零点。相对理论goal姿态的
    # 原始FK值在MOVE阶段通常为负，不适合直接解释为执行器运动距离。
    actual_closure_origin = None
    sent_closure_origin = None
    move_position_bias_l = np.zeros(7)
    move_position_bias_r = np.zeros(7)

    # GMO稳态偏置
    gmo_bias_F_l = np.zeros(3)
    gmo_bias_F_r = np.zeros(3)
    gmo_bias_samples_l = []
    gmo_bias_samples_r = []
    gmo_bias_applied = False
    bias_collect_start = args.move_time + args.grasp_time - 2.0
    tangent_bias_samples = []
    tangent_bias = 0.0
    tangent_bias_applied = False
    test_dynamic_state_reset = False
    recent_external_abs = deque(maxlen=10)  # 100 Hz下最近约100 ms
    test_external_abs_history = []
    test_external_peak_abs = 0.0
    test_external_peak_signed = 0.0

    def get_phase(exp_time):
        if exp_time < args.move_time:
            return 0
        elif exp_time < args.move_time + args.grasp_time:
            return 1
        else:
            return 2

    def run_one_step(exp_time, target_l, target_r,
                     move_integral_enabled=False):
        nonlocal last_time, last_print, record_count
        nonlocal gmo_bias_F_l, gmo_bias_F_r, gmo_bias_samples_l, gmo_bias_samples_r, gmo_bias_applied
        nonlocal last_target_l, last_target_r
        nonlocal actual_closure_origin
        nonlocal move_position_bias_l, move_position_bias_r
        nonlocal tangent_bias, tangent_bias_applied, test_dynamic_state_reset
        nonlocal test_external_peak_abs, test_external_peak_signed

        now = time.time()
        dt = now - last_time
        last_time = now
        phase = get_phase(exp_time)
        requested_target_l = target_l.copy()
        requested_target_r = target_r.copy()

        # 读取力传感器
        raw_force = force_reader.read_force()
        force_data.update(raw_force)
        f_left_6d = force_data.get_forces()
        F_I_est, f_left_normal = force_data.get_internal_force()
        F_E_sensor = force_data.get_external_force_from_sensor()

        # 关节状态
        cur = server.manager.get_current_arm_states()
        cur_l_q = np.array(cur["left_q"])
        cur_r_q = np.array(cur["right_q"])
        cur_l_dq = np.array(cur["left_dq"])
        cur_r_dq = np.array(cur["right_dq"])
        cur_l_tau = np.array(cur["left_tau"])
        cur_r_tau = np.array(cur["right_tau"])
        cur_l_dq = check_velocity_limits(cur_l_dq, dq_max)
        cur_r_dq = check_velocity_limits(cur_r_dq, dq_max)

        pb.update_states(cur_l_q, cur_r_q)
        if ik_actuator is not None:
            raw_actual_closure = ik_actuator.measure_total_closure(cur_l_q, cur_r_q)
            if phase == 0:
                actual_closure = np.nan
            else:
                if actual_closure_origin is None:
                    actual_closure_origin = raw_actual_closure
                    print(f"\n[IK] GRASP实际闭合量清零；相对理论goal的初始偏差="
                          f"{1000*raw_actual_closure:+.2f}mm")
                actual_closure = raw_actual_closure - actual_closure_origin
        else:
            actual_closure = np.nan

        # 左臂 GMO
        M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
        r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)
        J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
        F_l = np.linalg.pinv(J_l.T) @ r_l

        # 右臂 GMO
        M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
        r_r = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
        J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
        F_r = np.linalg.pinv(J_r.T) @ r_r

        # GMO稳态偏置处理
        if phase == 1 and exp_time >= bias_collect_start:
            gmo_bias_samples_l.append(F_l.copy())
            gmo_bias_samples_r.append(F_r.copy())
        elif phase == 2:
            if not gmo_bias_applied and len(gmo_bias_samples_l) > 10:
                gmo_bias_F_l = np.mean(gmo_bias_samples_l, axis=0)
                gmo_bias_F_r = np.mean(gmo_bias_samples_r, axis=0)
                gmo_bias_applied = True
                print(f"\n[GMO] 稳态偏置: L={gmo_bias_F_l}, R={gmo_bias_F_r}")
            if gmo_bias_applied:
                F_l = F_l - gmo_bias_F_l
                F_r = F_r - gmo_bias_F_r

        # GMO碰撞力
        gmo_force = (np.linalg.norm(F_l) + np.linalg.norm(F_r)) / 2.0

        # 碰撞外力采用GRASP末尾稳态切向力去偏后的值。力传感器最初零偏
        # 在MOVE前采集，姿态变化和物体重力会在切向通道引入新的稳态项。
        if phase == 1 and exp_time >= bias_collect_start:
            tangent_bias_samples.append(float(F_E_sensor))
        if phase == 2:
            if not tangent_bias_applied:
                if tangent_bias_samples:
                    tangent_bias = float(np.median(tangent_bias_samples))
                tangent_bias_applied = True
                print(f"\n[力传感器] TEST切向力稳态偏置: {tangent_bias:+.3f}N")
            if not test_dynamic_state_reset:
                # TEST从零滑移状态开始，避免GRASP期间的估计量污染碰撞结果。
                sifr.delta = 0.0
                sifr.delta_dot = 0.0
                test_dynamic_state_reset = True
            F_E = F_E_sensor - tangent_bias
            recent_external_abs.append(abs(F_E))
            test_external_abs_history.append(abs(F_E))
            if abs(F_E) > test_external_peak_abs:
                test_external_peak_abs = abs(F_E)
                test_external_peak_signed = F_E
        else:
            # GRASP阶段只完成标称内力跟踪，不把稳态切向载荷当作碰撞。
            F_E = 0.0

        # SIFR控制器（期望内力固定）
        if phase >= 1:
            F_I_des, delta_roll, delta, fc, active = sifr.update(F_E, F_I_est, dt)
            if args.actuation_mode == 'ik':
                target_l, target_r, _ = ik_actuator.targets(delta_roll)
            else:
                target_l[1] -= args.roll_action_sign * delta_roll / 2.0
                target_r[1] += args.roll_action_sign * delta_roll / 2.0
        else:
            F_I_des = args.F_fixed
            delta_roll = 0.0
            delta = 0.0
            fc = 0.0
            active = False

        # 纯位置外环积分：只在MOVE最终目标保持阶段学习静差补偿。
        # GRASP/TEST沿用该位置偏置，但motor_cmd.tau始终保持为零。
        if phase == 0 and move_integral_enabled:
            move_position_bias_l = update_position_bias(
                move_position_bias_l, requested_target_l - cur_l_q, dt,
                args.move_position_ki, args.move_position_bias_max,
                args.move_position_deadband)
            move_position_bias_r = update_position_bias(
                move_position_bias_r, requested_target_r - cur_r_q, dt,
                args.move_position_ki, args.move_position_bias_max,
                args.move_position_deadband)
            target_l = requested_target_l + move_position_bias_l
            target_r = requested_target_r + move_position_bias_r
        elif phase >= 1:
            target_l = target_l + move_position_bias_l
            target_r = target_r + move_position_bias_r

        # 本帧希望发送的纯位置命令（含外环偏置）。request_gap只统计
        # 后续关节限位或速度限制造成的未发送部分。
        command_request_l = target_l.copy()
        command_request_r = target_r.copy()

        # 关节角限位
        target_l = check_joint_limits(target_l, q_min, q_max)
        target_r = check_joint_limits(target_r, q_min, q_max)

        # 关节角增量限幅
        delta_l = target_l - last_target_l
        delta_r = target_r - last_target_r
        frame_limit = joint_delta_limit_for_phase(
            phase, args.actuation_mode,
            args.joint_delta_max, args.ik_joint_delta_max,
            dt, args.move_joint_speed_max)
        delta_l = np.clip(delta_l, -frame_limit, frame_limit)
        delta_r = np.clip(delta_r, -frame_limit, frame_limit)
        target_l = last_target_l + delta_l
        target_r = last_target_r + delta_r
        request_gap = max(
            float(np.max(np.abs(command_request_l - target_l))),
            float(np.max(np.abs(command_request_r - target_r))))
        last_target_l = target_l.copy()
        last_target_r = target_r.copy()
        left_goal_rmse = float(np.sqrt(np.mean(
            (requested_target_l - cur_l_q)**2)))
        right_goal_rmse = float(np.sqrt(np.mean(
            (requested_target_r - cur_r_q)**2)))
        left_tracking_rmse = float(np.sqrt(np.mean((target_l - cur_l_q)**2)))
        right_tracking_rmse = float(np.sqrt(np.mean((target_r - cur_r_q)**2)))
        if ik_actuator is not None and sent_closure_origin is not None:
            sent_target_closure = (
                ik_actuator.measure_total_closure(target_l, target_r) -
                sent_closure_origin)
        else:
            sent_target_closure = np.nan

        # 发送控制指令
        server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
                                     [0.0]*7, [0.0]*7)


        # 记录
        row = np.concatenate([
            [exp_time, phase],
            cur_l_q, cur_r_q,
            F_l, F_r,
            [f_left_normal],
            [F_I_est, F_E, F_E_sensor],
            [F_I_des, delta, delta_roll, fc, 1.0 if active else 0.0],
            f_left_6d,
            [actual_closure,
             action_max if args.actuation_mode == 'ik' else np.nan,
             left_tracking_rmse, right_tracking_rmse, sent_target_closure]
        ])
        data_file.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        record_count += 1

        # 打印
        if exp_time - last_print >= 0.1:
            phase_str = ["MOVE", "GRASP", "TEST"][phase]
            fc_str = f"fc={fc:.2f}" + ("!" if fc > 1.0 else "")
            action_text = (f"MOVE loop={1.0/max(dt, 1e-6):.1f}Hz "
                           f"step_limit={frame_limit:.4f}rad "
                           f"request_gap={request_gap:.4f}rad "
                           f"I-bias={max(np.max(np.abs(move_position_bias_l)), np.max(np.abs(move_position_bias_r))):.4f}rad "
                           f"goal_err={np.degrees(left_goal_rmse):.2f}/"
                           f"{np.degrees(right_goal_rmse):.2f}deg "
                           f"cmd_track={np.degrees(left_tracking_rmse):.2f}/"
                           f"{np.degrees(right_tracking_rmse):.2f}deg"
                           if phase == 0 else
                           f"closure cmd/sent/actual/limit="
                           f"{1000*delta_roll:.1f}/{1000*sent_target_closure:.1f}/"
                           f"{1000*actual_closure:.1f}/"
                           f"{1000*action_max:.1f}mm "
                           f"lag={1000*(sent_target_closure-actual_closure):+.1f}mm "
                           f"sat={'Y' if delta_roll >= 0.995*action_max else 'N'} "
                           f"qerr={np.degrees(left_tracking_rmse):.2f}/"
                           f"{np.degrees(right_tracking_rmse):.2f}deg"
                           if args.actuation_mode == 'ik' else
                           f"delta_roll={delta_roll:+.4f}rad")
            impact_text = (f"Fext={F_E:+.2f}N "
                           f"mean100ms={np.mean(recent_external_abs):.2f}N "
                           f"peak100ms={max(recent_external_abs, default=0.0):.2f}N "
                           f"TESTpeak={test_external_peak_abs:.2f}N"
                           if phase == 2 else "Fext=waiting-for-TEST")
            print(f"\r  t={exp_time:5.2f}s [{phase_str}] [SIFR-fixed] | "
                  f"GMO={gmo_force:5.1f}N F_E={F_E:5.1f}N | "
                  f"F_I={F_I_des:5.1f}(fixed) est={F_I_est:5.1f} | "
                  f"δ={delta:+.4f} {fc_str} | "
                  f"{action_text} | {impact_text} | "
                  f"左传感器 Fn={f_left_normal:5.1f} Ft={F_E_sensor:+5.1f}",
                  end="", flush=True)
            last_print = exp_time

    try:
        # 阶段1: MOVE
        print(f"\n[阶段1] 运动到预备姿态 ({args.move_time}s)...")
        move_start = time.time()
        while True:
            elapsed = time.time() - move_start
            if elapsed >= args.move_time:
                break
            target_l = quintic_interpolate(
                start_l, goal_l, elapsed, args.move_time)
            target_r = quintic_interpolate(
                start_r, goal_r, elapsed, args.move_time)
            run_one_step(elapsed, target_l.copy(), target_r.copy())
            time.sleep(t_step)

        # 仍使用与GRASP/TEST相同的run_one_step下发链路。先在最终目标
        # 上保持move_settle_time，再进入额外的到位超时判断；旧时序把
        # settle放在验收之后，导致用户设置的稳定时间根本没有执行。
        move_phase_time = max(0.0, args.move_time - 1e-6)
        if args.move_settle_time > 0.0:
            print(f"\n[阶段1] 保持最终目标稳定 {args.move_settle_time:.2f}s...")
            settle_start = time.time()
            while time.time() - settle_start < args.move_settle_time:
                run_one_step(move_phase_time, goal_l.copy(), goal_r.copy(),
                             move_integral_enabled=True)
                time.sleep(t_step)

        convergence_start = time.time()
        while True:
            run_one_step(move_phase_time, goal_l.copy(), goal_r.copy(),
                         move_integral_enabled=True)
            current_move = server.manager.get_current_arm_states()
            move_error_l, move_joint_l, move_signed_l = joint_error_details(
                current_move["left_q"], goal_l)
            move_error_r, move_joint_r, move_signed_r = joint_error_details(
                current_move["right_q"], goal_r)
            if max(move_error_l, move_error_r) <= args.move_goal_tolerance:
                break
            if time.time() - convergence_start >= args.move_convergence_timeout:
                raise RuntimeError(
                    f"MOVE到位超时: L {move_joint_l}={move_signed_l:+.4f}rad, "
                    f"R {move_joint_r}={move_signed_r:+.4f}rad；"
                    f"最大I-bias={max(np.max(np.abs(move_position_bias_l)), np.max(np.abs(move_position_bias_r))):.4f}rad；"
                    "request_gap为0时表示含补偿的位置命令已完整发送")
            time.sleep(t_step)

        # 补偿额外到位等待，使GRASP的实验时间仍从move_time开始。
        exp_start = time.time() - args.move_time
        rebase_start = time.time()
        print(f"\n[阶段1] 已通过实验主循环到达预备姿态")

        # 锁定MOVE结束后的真机实际姿态，作为GRASP和TEST共同参考零位。
        reference = server.manager.get_current_arm_states()
        hold_l = np.array(reference["left_q"])
        hold_r = np.array(reference["right_q"])
        move_error_l, move_joint_l, move_signed_l = joint_error_details(
            hold_l, goal_l)
        move_error_r, move_joint_r, move_signed_r = joint_error_details(
            hold_r, goal_r)
        print(f"[阶段1] 最大关节到位误差: "
              f"L {move_joint_l}={move_signed_l:+.4f}rad, "
              f"R {move_joint_r}={move_signed_r:+.4f}rad")
        if max(move_error_l, move_error_r) > args.move_goal_tolerance:
            raise RuntimeError(
                "MOVE实际姿态未到达指定位置，拒绝用错误姿态建立IK零点；"
                "请检查底层控制权、关节跟踪或适当增加--move-settle-time")
        print(f"[阶段1] 学得纯位置偏置: "
              f"L={move_position_bias_l.tolist()}, "
              f"R={move_position_bias_r.tolist()}")
        sifr.reset()
        if args.actuation_mode == 'ik':
            ik_actuator = SymmetricNormalIKActuator(
                pb, hold_l, hold_r,
                max_displacement=args.ik_max_displacement,
                damping=args.ik_damping,
                iterations=args.ik_iterations,
                max_step=args.ik_max_step,
                q_min=q_min, q_max=q_max)
            actual_closure_origin = ik_actuator.measure_total_closure(hold_l, hold_r)
            biased_hold_l = check_joint_limits(
                hold_l + move_position_bias_l, q_min, q_max)
            biased_hold_r = check_joint_limits(
                hold_r + move_position_bias_r, q_min, q_max)
            sent_closure_origin = ik_actuator.measure_total_closure(
                biased_hold_l, biased_hold_r)
        data_file.write("# actual_reference_left_q: " +
                        " ".join(f"{v:.8f}" for v in hold_l) + "\n")
        data_file.write("# actual_reference_right_q: " +
                        " ".join(f"{v:.8f}" for v in hold_r) + "\n")
        print("[参考零位] 已使用MOVE结束真机姿态重建控制参考；"
              "cmd/sent/actual从同一零点开始")
        # IK求解属于阶段切换准备，不计入GRASP时长，也不能形成一个大dt。
        exp_start += time.time() - rebase_start
        last_time = time.time()

        # 阶段2: GRASP
        print(f"\n[阶段2] 保持姿态，GMO去偏采集 ({args.grasp_time}s)...")
        grasp_start = time.time()
        while True:
            elapsed = time.time() - grasp_start
            exp_time = time.time() - exp_start
            if elapsed >= args.grasp_time:
                break
            run_one_step(exp_time, hold_l.copy(), hold_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段2] 完成")

        # 阶段3: TEST
        print(f"\n[阶段3] 碰撞测试 ({args.test_time}s) — 可施加外部冲击，期望内力保持固定")
        test_start = time.time()
        while True:
            elapsed = time.time() - test_start
            exp_time = time.time() - exp_start
            if elapsed >= args.test_time:
                break
            run_one_step(exp_time, hold_l.copy(), hold_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段3] 测试完成")
        smooth_peak = (float(np.max(np.convolve(
            np.asarray(test_external_abs_history), np.ones(5)/5.0, mode='valid')))
            if len(test_external_abs_history) >= 5 else test_external_peak_abs)
        print(f"[碰撞统计] TEST峰值外力={test_external_peak_abs:.3f}N "
              f"(signed={test_external_peak_signed:+.3f}N), "
              f"50ms平滑峰值={smooth_peak:.3f}N")

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断")
    except Exception as e:
        print(f"\n\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 80)
        print("  实验结束，清理中...")
        print("=" * 80)

        data_file.write(f"# 记录总帧数: {record_count}\n")
        data_file.close()
        print(f"[数据] 已保存 {record_count} 帧到: {output_path}")

        try:
            cur = server.manager.get_current_arm_states()
            # server.manager.set_arm_poses(cur["left_q"], cur["right_q"], [0.0]*7, [0.0]*7)
            server.stop()
            print("[G1] 已停止")
        except Exception as e:
            print(f"[G1] 停止异常: {e}")

        force_reader.close()
        print("[力传感器] 已断开")

        pb.disconnect()
        print("[PyBullet] 已断开")
        print(f"\n[完成] 数据文件: {output_path}")


if __name__ == "__main__":
    main()
