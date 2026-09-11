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
                 roll_offset_max=0.10, force_deadband=0.2,
                 roll_offset_min=None):
        self.F_fixed = F_fixed
        self.mu = mu
        self.m_object = m_object
        self.K_f = K_f
        self.roll_offset_max = roll_offset_max
        self.roll_offset_min = (-roll_offset_max if roll_offset_min is None
                                else roll_offset_min)
        self.force_deadband = force_deadband
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
        net_force = force_demand - self.mu * F_I_des
        if self.delta_dot <= 0.0 and net_force <= 0.0:
            self.delta_dot = 0.0  # 静摩擦区：不允许“反向滑移”伪影
        else:
            self.delta_dot = max(0.0, self.delta_dot + net_force / self.m_object * dt)
        self.delta += self.delta_dot * dt
        # 摩擦锥裕度
        fc = force_demand / max(self.mu * F_I_des, 1e-6)

        # 简化力积分器：累计肩roll位置偏移，直到实测内力跟上期望值。
        force_error = F_I_des - F_I_est
        if abs(force_error) < self.force_deadband:
            force_error = 0.0
        self.roll_offset += self.K_f * force_error * dt
        self.roll_offset = np.clip(
            self.roll_offset, self.roll_offset_min, self.roll_offset_max)
        delta_roll = self.roll_offset

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


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="SIFR双臂协作搬运实验（力传感器版本，对比实验）")
    parser.add_argument('--left_q', type=float, nargs=7, required=True)
    parser.add_argument('--right_q', type=float, nargs=7, required=True)
    parser.add_argument('--move-time', type=float, default=10.0)
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
        if not (0.0 < args.ik_max_displacement <= 0.015):
            parser.error('--ik-max-displacement必须在(0, 0.015] m内；更大位移需先重新仿真验证')
        if args.ik_force_gain <= 0.0 or args.ik_joint_delta_max <= 0.0:
            parser.error('--ik-force-gain和--ik-joint-delta-max必须为正数')

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
    sifr = SIFRController(F_fixed=args.F_fixed, mu=args.mu,
                           m_object=args.m_object, K_f=force_gain,
                           roll_offset_max=action_max,
                           force_deadband=args.force_deadband,
                           roll_offset_min=(0.0 if args.actuation_mode == 'ik' else None))

    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)
    ik_actuator = None
    if args.actuation_mode == 'ik':
        ik_actuator = SymmetricNormalIKActuator(
            pb, goal_l, goal_r,
            max_displacement=args.ik_max_displacement,
            damping=args.ik_damping,
            iterations=args.ik_iterations,
            max_step=args.ik_max_step,
            q_min=q_min, q_max=q_max)

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
    data_file.write("# 列: time phase(0=MOVE,1=GRASP,2=TEST) "
                     "left_q(7) right_q(7) "
                     "left_gmo_F(3) right_gmo_F(3) "
                     "force_left_normal "
                     "F_I_est(内力估计,N) F_E_gmo(GMO外力,N) F_E_sensor(传感器外力,N) "
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

    # GMO稳态偏置
    gmo_bias_F_l = np.zeros(3)
    gmo_bias_F_r = np.zeros(3)
    gmo_bias_samples_l = []
    gmo_bias_samples_r = []
    gmo_bias_applied = False
    bias_collect_start = args.move_time + args.grasp_time - 2.0

    def get_phase(exp_time):
        if exp_time < args.move_time:
            return 0
        elif exp_time < args.move_time + args.grasp_time:
            return 1
        else:
            return 2

    def run_one_step(exp_time, target_l, target_r):
        nonlocal last_time, last_print, record_count
        nonlocal gmo_bias_F_l, gmo_bias_F_r, gmo_bias_samples_l, gmo_bias_samples_r, gmo_bias_applied
        nonlocal last_target_l, last_target_r
        nonlocal actual_closure_origin

        now = time.time()
        dt = now - last_time
        last_time = now
        phase = get_phase(exp_time)

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

        # 估计外力F_E
        # 公式(39)-(41)使用左手六维力传感器的有符号切向力。
        F_E = F_E_sensor

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

        # 关节角限位
        target_l = check_joint_limits(target_l, q_min, q_max)
        target_r = check_joint_limits(target_r, q_min, q_max)

        # 关节角增量限幅
        delta_l = target_l - last_target_l
        delta_r = target_r - last_target_r
        frame_limit = (min(args.joint_delta_max, args.ik_joint_delta_max)
                       if args.actuation_mode == 'ik' else args.joint_delta_max)
        delta_l = np.clip(delta_l, -frame_limit, frame_limit)
        delta_r = np.clip(delta_r, -frame_limit, frame_limit)
        target_l = last_target_l + delta_l
        target_r = last_target_r + delta_r
        last_target_l = target_l.copy()
        last_target_r = target_r.copy()
        left_tracking_rmse = float(np.sqrt(np.mean((target_l - cur_l_q)**2)))
        right_tracking_rmse = float(np.sqrt(np.mean((target_r - cur_r_q)**2)))
        if ik_actuator is not None and actual_closure_origin is not None:
            sent_target_closure = (
                ik_actuator.measure_total_closure(target_l, target_r) -
                actual_closure_origin)
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
            action_text = (f"closure cmd/sent/actual/limit="
                           f"{1000*delta_roll:.1f}/{1000*sent_target_closure:.1f}/"
                           f"{1000*actual_closure:.1f}/"
                           f"{1000*action_max:.1f}mm "
                           f"sat={'Y' if delta_roll >= 0.995*action_max else 'N'} "
                           f"qerr={np.degrees(left_tracking_rmse):.2f}/"
                           f"{np.degrees(right_tracking_rmse):.2f}deg"
                           if args.actuation_mode == 'ik' else
                           f"delta_roll={delta_roll:+.4f}rad")
            print(f"\r  t={exp_time:5.2f}s [{phase_str}] [SIFR-fixed] | "
                  f"GMO={gmo_force:5.1f}N F_E={F_E:5.1f}N | "
                  f"F_I={F_I_des:5.1f}(fixed) est={F_I_est:5.1f} | "
                  f"δ={delta:+.4f} {fc_str} | "
                  f"{action_text} | "
                  f"左传感器 Fn={f_left_normal:5.1f} Ft={F_E_sensor:+5.1f}",
                  end="", flush=True)
            last_print = exp_time

    try:
        # 阶段1: MOVE
        print(f"\n[阶段1] 运动到预备姿态 ({args.move_time}s)...")
        move_start = time.time()
        while True:
            elapsed = time.time() - move_start
            exp_time = time.time() - exp_start
            if elapsed >= args.move_time:
                break
            target_l = quintic_interpolate(start_l, goal_l, elapsed, args.move_time)
            target_r = quintic_interpolate(start_r, goal_r, elapsed, args.move_time)
            run_one_step(exp_time, target_l.copy(), target_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段1] 到达预备姿态")

        # 阶段2: GRASP
        print(f"\n[阶段2] 保持姿态，GMO去偏采集 ({args.grasp_time}s)...")
        grasp_start = time.time()
        while True:
            elapsed = time.time() - grasp_start
            exp_time = time.time() - exp_start
            if elapsed >= args.grasp_time:
                break
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
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
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段3] 测试完成")

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
