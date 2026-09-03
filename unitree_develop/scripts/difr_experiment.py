#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-03 (v2: 公式41 QP解析解 + 三阶段实验)
Description: 双臂协作搬运实验 — DIFR (Dynamic Internal Force Regulation)
             论文方法：基于滑动动力学的QP问题(公式41)动态调节期望内力

实验三阶段：
  阶段1 MOVE  (0~10s): 双臂五次多项式运动到搬运预备姿态
  阶段2 GRASP (10~20s): 夹持物体，通过肩关节roll微调跟踪标称期望内力F_0
  阶段3 TEST  (20~50s): 碰撞测试，DIFR激活动态调节期望内力，验证公式41

DIFR核心算法（公式41 QP解析解）：
  滑动动力学(公式39-40):
    δ̇(k) = δ̇(k-1) + (F_E - μ·F_I) / m_o · Δt
    δ(k)  = δ(k-1) + δ̇(k) · Δt

  QP问题(公式41):
    min  α0·(F_I - F_0)² + α1·δ²
    s.t. 滑动动力学, δ_min≤δ≤δ_max, F_I,min≤F_I≤F_I,max

  1维解析解:
    k = μ·Δt²/m_o,  C = δ_prev + δ̇_prev·Δt + F_E/m_o·Δt²
    F_I* = (α0·F_0 + α1·k·C) / (α0 + α1·k²)
    F_I,d = clip(F_I*, F_I,min, F_I,max)

  位置控制下内力跟踪: Δφ = K_f·(F_I,d - F_I,est)
    左肩roll += Δφ/2, 右肩roll -= Δφ/2

用法：
  python difr_experiment.py \
      --left_q  -0.8 0.7 -0.7 0.4 0.0 -0.6 0.0 \
      --right_q  0.8 0.7  0.7 0.4 0.0  0.6 0.0 \
      --F0 10.0 --F-min 5.0 --F-max 30.0 --mu 0.3 --m-object 1.0
'''
import os
import sys
import time
import argparse
import threading
import multiprocessing as mp
import numpy as np
import pybullet as p
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server


# ================================================================
# 触觉子进程（spawn 隔离 DDS）
# ================================================================
def tactile_subscriber_worker(data_queue, log_queue, stop_event):
    try:
        import rclpy
        from rclpy.node import Node
        from ros2_stark_msgs.msg import TouchStatus
    except ImportError as e:
        log_queue.put(f"[触觉] 导入失败: {e}")
        return
    FINGER_COUNT = 5
    class TactileNode(Node):
        def __init__(self):
            super().__init__("difr_tactile_sub")
            self.left_sub = self.create_subscription(TouchStatus, "/left_hand/touch_status_126", self.left_cb, 10)
            self.right_sub = self.create_subscription(TouchStatus, "/right_hand/touch_status_127", self.right_cb, 10)
            log_queue.put("[触觉] 订阅已创建")
        def _extract(self, msg):
            nf1 = np.zeros(FINGER_COUNT)
            for i in range(min(FINGER_COUNT, len(msg.data))):
                nf1[i] = float(msg.data[i].normal_force1)
            return nf1
        def left_cb(self, msg):
            data_queue.put(("left", self._extract(msg).tolist(), time.time()))
        def right_cb(self, msg):
            data_queue.put(("right", self._extract(msg).tolist(), time.time()))
    try:
        rclpy.init()
        node = TactileNode()
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        log_queue.put(f"[触觉] 错误: {e}")


# ================================================================
# 触觉数据容器
# ================================================================
class TactileData:
    def __init__(self):
        self.left_nf1 = np.zeros(5)
        self.right_nf1 = np.zeros(5)
        self._lock = threading.Lock()
    def update(self, hand, nf1):
        with self._lock:
            if hand == "left":
                self.left_nf1 = np.array(nf1, dtype=float)/1000
            else:
                self.right_nf1 = np.array(nf1, dtype=float)/1000
    def get_sums(self):
        with self._lock:
            return float(np.sum(self.left_nf1)/1000), float(np.sum(self.right_nf1)/1000)
    def get_fingers(self):
        with self._lock:
            return self.left_nf1.copy()/1000, self.right_nf1.copy()/1000


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
# 手自重补偿器
# ================================================================
class HandGravityCompensator:
    def __init__(self, hand_mass=0.45, g=9.81):
        self.F_hand = np.array([0.0, 0.0, -hand_mass * g])
    def compensate(self, r_gmo, J_p):
        tau_hand = J_p.T @ self.F_hand
        return r_gmo - tau_hand, tau_hand


# ================================================================
# DIFR 控制器（论文公式41 QP解析解）
# ================================================================
class DIFRController:
    '''
    Dynamic Internal Force Regulation — 基于滑动动力学的QP问题(公式41)

    决策变量: F_I (期望内力，标量，法向夹持力)
    目标: min α0·(F_I - F_0)² + α1·δ²
    约束: 滑动动力学(公式39-40), δ限位, F_I限位

    1维解析解:
      k = μ·Δt²/m_o
      C = δ_prev + δ̇_prev·Δt + F_E/m_o·Δt²
      F_I* = (α0·F_0 + α1·k·C) / (α0 + α1·k²)
    '''
    def __init__(self, F_0=16.35, F_min=5.0, F_max=40.0,
                 mu=0.3, m_object=1.0, alpha0=1.0, alpha1=100.0,
                 delta_min=-0.05, delta_max=0.05,
                 collision_threshold=8.0, fc_threshold=0.8, K_f=0.002):
        # QP参数
        self.F_0 = F_0              # 标称内力（物理标定: m*g/(2*mu)）
        self.F_min = F_min          # 内力下限
        self.F_max = F_max          # 内力上限
        self.mu = mu                 # 摩擦系数
        self.m_object = m_object     # 物体质量
        self.alpha0 = alpha0         # 内力偏离标称值的权重
        self.alpha1 = alpha1         # 滑动位移的权重（越大越积极抑制滑动）
        self.delta_min = delta_min   # 滑动位移下限
        self.delta_max = delta_max   # 滑动位移上限
        self.collision_threshold = collision_threshold
        self.fc_threshold = fc_threshold  # 摩擦锥裕度激活阈值
        self.K_f = K_f               # 内力→位置增益

        # 状态
        self.delta = 0.0             # 滑动位移 δ
        self.delta_dot = 0.0         # 滑动速度 δ̇
        self.F_I_des = F_0           # 当前期望内力
        self.active = False           # DIFR是否激活
        self.F_E_last = 0.0          # 上一帧外力估计

    def reset(self):
        self.delta = 0.0
        self.delta_dot = 0.0
        self.F_I_des = self.F_0
        self.active = False

    def update(self, F_E, F_I_est, gmo_force_norm, dt):
        '''
        DIFR更新 — 解公式41的QP问题
        Args:
            F_E: 估计的外部切向力（导致滑动的力，从GMO估计）
            F_I_est: 估计的实际内力（从触觉传感器）
            gmo_force_norm: GMO估计外力范数（碰撞检测）
            dt: 时间步长
        Returns:
            F_I_des: 更新后的期望内力
            delta_roll: 肩关节roll偏移量（位置控制下实现内力跟踪）
            delta: 滑动位移估计
            fc: 摩擦锥裕度 = F_E / (mu * F_I_des)，fc>1表示即将滑动
            active: DIFR是否激活
        '''
        # 摩擦锥裕度（用当前期望内力估算）
        fc = F_E / max(self.mu * self.F_I_des, 1e-6)

        # 激活条件：碰撞检测 或 滑动位移/速度过大 或 摩擦锥裕度接近1
        collision = gmo_force_norm > self.collision_threshold
        slip = abs(self.delta) > 0.005 or abs(self.delta_dot) > 0.02
        friction_margin_low = fc > self.fc_threshold
        self.active = collision or slip or friction_margin_low

        if self.active:
            # ===== 公式41 QP解析解 =====
            # 滑动动力学预测: δ̇ = δ̇_prev + (F_E - μF_I)/m_o · Δt
            #                  δ = δ_prev + δ̇ · Δt
            # 代入目标: min α0(F_I-F_0)² + α1 δ²
            k = self.mu * dt**2 / self.m_object
            C = self.delta + self.delta_dot * dt + F_E / self.m_object * dt**2

            # 无约束最优解
            denom = self.alpha0 + self.alpha1 * k**2
            F_I_star = (self.alpha0 * self.F_0 + self.alpha1 * k * C) / denom

            # 投影到内力限位
            F_I_des = np.clip(F_I_star, self.F_min, self.F_max)

            # 用优化后的F_I更新滑动状态（公式39-40）
            self.delta_dot = self.delta_dot + (F_E - self.mu * F_I_des) / self.m_object * dt
            self.delta = self.delta + self.delta_dot * dt

            # 滑动位移限位：如果超限，强制调整F_I
            if self.delta > self.delta_max:
                F_I_des = min(self.F_max, F_I_des + 3.0)
                self.delta = self.delta_max
                self.delta_dot = 0.0
            elif self.delta < self.delta_min:
                F_I_des = max(self.F_min, F_I_des - 3.0)
                self.delta = self.delta_min
                self.delta_dot = 0.0
        else:
            # 未激活：保持标称内力，滑动状态衰减
            F_I_des = self.F_0
            self.delta *= 0.95
            self.delta_dot *= 0.95

        self.F_I_des = F_I_des
        self.F_E_last = F_E

        # 更新摩擦锥裕度（用优化后的F_I_des）
        fc = F_E / max(self.mu * F_I_des, 1e-6)

        # 位置控制下的内力跟踪：肩关节roll相对偏移
        # 增大F_I_des → 增大夹持 → 左肩roll+，右肩roll-
        delta_roll = self.K_f * (F_I_des - F_I_est)
        delta_roll = np.clip(delta_roll, -0.05, 0.05)

        return F_I_des, delta_roll, self.delta, fc, self.active


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
        self.left_ee = self.left_joints[-1] + 1
        self.right_ee = self.right_joints[-1] + 1
        self.left_pos = [self.joint_to_pos[jid] for jid in self.left_joints]
        self.right_pos = [self.joint_to_pos[jid] for jid in self.right_joints]
        print(f"[PyBullet] 左臂: {self.left_joints}, 右臂: {self.right_joints}")

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

def drain_queue(queue):
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except Exception:
            break
    return items

def check_joint_limits(q, q_min, q_max):
    return np.clip(q, q_min, q_max)

def check_velocity_limits(dq, dq_max=2.0):
    return np.clip(dq, -dq_max, dq_max)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="DIFR双臂协作搬运实验（公式41 QP）")
    parser.add_argument('--left_q', type=float, nargs=7, required=True)
    parser.add_argument('--right_q', type=float, nargs=7, required=True)
    parser.add_argument('--move-time', type=float, default=10.0, help='阶段1运动时间')
    parser.add_argument('--grasp-time', type=float, default=10.0, help='阶段2夹持时间')
    parser.add_argument('--test-time', type=float, default=30.0, help='阶段3碰撞测试时间')
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--gmo-gain', type=float, default=5.0)
    parser.add_argument('--hand-mass', type=float, default=0.45)
    # DIFR QP参数（物理标定：1kg物体，μ=0.3 → F_0 = m*g/(2*μ) ≈ 16.35N/臂）
    parser.add_argument('--F0', type=float, default=16.35, help='标称期望内力F_0 (N/臂), 默认16.35=1kg*9.81/(2*0.3)')
    parser.add_argument('--F-min', type=float, default=5.0, help='内力下限')
    parser.add_argument('--F-max', type=float, default=25.0, help='内力上限')
    parser.add_argument('--mu', type=float, default=0.3, help='摩擦系数')
    parser.add_argument('--m-object', type=float, default=1.0, help='物体质量(kg)')
    parser.add_argument('--alpha0', type=float, default=1.0, help='QP内力偏离权重')
    parser.add_argument('--alpha1', type=float, default=100.0, help='QP滑动位移权重')
    parser.add_argument('--fc-threshold', type=float, default=0.8, help='摩擦锥裕度激活阈值')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--urdf', type=str, default='description/g1_14dof_brainco_hand.urdf')
    args = parser.parse_args()

    total_time = args.move_time + args.grasp_time + args.test_time
    print("=" * 80)
    print("  DIFR 双臂协作搬运实验 (Dynamic Internal Force Regulation, Eq.41 QP)")
    print("=" * 80)
    print(f"  阶段1 MOVE:  {args.move_time}s (运动到预备姿态)")
    print(f"  阶段2 GRASP: {args.grasp_time}s (夹持物体，跟踪F_0={args.F0}N)")
    print(f"  阶段3 TEST:  {args.test_time}s (碰撞测试，DIFR激活)")
    print(f"  QP参数: F_0={args.F0}, F∈[{args.F_min},{args.F_max}], μ={args.mu}, m_o={args.m_object}")
    print(f"  权重: α0={args.alpha0}, α1={args.alpha1}")
    print("=" * 80)

    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"difr_experiment_{ts}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    q_min = np.array([-2.0, -1.2, -2.5, -0.7, -1.6, -1.4, -1.2])
    q_max = np.array([ 2.0,  1.2,  2.5,  1.8,  1.6,  1.4,  1.2])
    dq_max = 0.2

    pb = DualArmPyBullet(args.urdf)
    gmo_left = GMOObserver(7, args.gmo_gain)
    gmo_right = GMOObserver(7, args.gmo_gain)
    hand_comp_left = HandGravityCompensator(args.hand_mass)
    hand_comp_right = HandGravityCompensator(args.hand_mass)
    difr = DIFRController(F_0=args.F0, F_min=args.F_min, F_max=args.F_max,
                           mu=args.mu, m_object=args.m_object,
                           alpha0=args.alpha0, alpha1=args.alpha1,
                           fc_threshold=args.fc_threshold)

    print(f"\n[初始化] 连接 G1...")
    server = G1Server(network_interface=args.interface, shm_name="dev/shm/6_axis_force_shm")
    server.start()
    time.sleep(0.5)
    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    tactile_data = TactileData()
    ctx = mp.get_context('spawn')
    data_queue = ctx.Queue(maxsize=200)
    log_queue = ctx.Queue(maxsize=50)
    stop_event = ctx.Event()
    tactile_process = ctx.Process(target=tactile_subscriber_worker,
                                   args=(data_queue, log_queue, stop_event), daemon=True)
    tactile_process.start()
    print(f"[触觉] 子进程 PID: {tactile_process.pid}")
    time.sleep(3.0)
    for msg in drain_queue(log_queue):
        print(f"  {msg}")

    data_file = open(output_path, 'w')
    data_file.write("# DIFR双臂协作搬运实验数据（公式41 QP）\n")
    data_file.write(f"# 方法: DIFR (Dynamic Internal Force Regulation)\n")
    data_file.write(f"# 时间: {datetime.now()}\n")
    data_file.write(f"# 阶段: MOVE={args.move_time}s, GRASP={args.grasp_time}s, TEST={args.test_time}s\n")
    data_file.write(f"# QP参数: F_0={args.F0}, F_min={args.F_min}, F_max={args.F_max}, mu={args.mu}, m_object={args.m_object}\n")
    data_file.write(f"# 权重: alpha0={args.alpha0}, alpha1={args.alpha1}\n")
    data_file.write("# 列: time phase(0=MOVE,1=GRASP,2=TEST) "
                     "left_q(7) right_q(7) left_dq(7) right_dq(7) "
                     "left_gmo_F(3) right_gmo_F(3) "
                     "tactile_left tactile_right "
                     "F_E(估计外力) F_I_des(期望内力) F_I_est(实际内力) "
                     "delta(滑动位移) delta_roll(肩roll偏移) fc(摩擦锥裕度=F_E/(mu*F_I)) difr_active(是否激活) "
                     "left_tactile_5f(5) right_tactile_5f(5)\n")
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
    difr.reset()

    def get_phase(exp_time):
        if exp_time < args.move_time:
            return 0  # MOVE
        elif exp_time < args.move_time + args.grasp_time:
            return 1  # GRASP
        else:
            return 2  # TEST

    def run_one_step(exp_time, target_l, target_r):
        nonlocal last_time, last_print, record_count

        now = time.time()
        dt = now - last_time
        last_time = now
        phase = get_phase(exp_time)

        # 触觉
        for item in drain_queue(data_queue):
            hand, nf1, ts = item
            tactile_data.update(hand, nf1)
        for msg in drain_queue(log_queue):
            print(f"\n  [触觉] {msg}")

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

        # 左臂 GMO + 手补偿
        if phase == 0:
            M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
            r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)
            J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
            F_l = np.linalg.pinv(J_l.T) @ r_l
        else:
            M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
            r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)

            J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
            r_l, _ = hand_comp_left.compensate(r_l, J_l)
            F_l = np.linalg.pinv(J_l.T) @ r_l

        # 右臂 GMO + 手补偿
        if phase == 0:
            M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
            r_r = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
            J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
            F_r = np.linalg.pinv(J_r.T) @ r_r
        else:
            M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
            r_r_raw = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
            J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
            r_r, _ = hand_comp_right.compensate(r_r_raw, J_r)
            F_r = np.linalg.pinv(J_r.T) @ r_r

        # 触觉力
        t_left, t_right = tactile_data.get_sums()
        f_left_5, f_right_5 = tactile_data.get_fingers()

        # GMO碰撞力（双臂平均）
        gmo_force = (np.linalg.norm(F_l) + np.linalg.norm(F_r)) / 2.0

        # 估计外力F_E（切向，导致滑动的力）
        # 简化：取GMO估计外力的水平分量（z方向主要是重力）
        F_E = np.sqrt(F_l[0]**2 + F_l[1]**2 + F_r[0]**2 + F_r[1]**2) / 2.0

        # 实际内力估计（左右触觉力差的绝对值，简化）
        F_I_est = abs(t_left - t_right) * 0.5

        # DIFR控制器（阶段2和阶段3生效）
        if phase >= 1:
            F_I_des, delta_roll, delta, fc, difr_active = difr.update(
                F_E, F_I_est, gmo_force, dt)
            # 应用肩关节roll偏移调节内力
            target_l[1] += delta_roll / 2.0
            target_r[1] -= delta_roll / 2.0
        else:
            F_I_des = args.F0
            delta_roll = 0.0
            delta = 0.0
            fc = 0.0
            difr_active = False

        # 关节角限位
        target_l = check_joint_limits(target_l, q_min, q_max)
        target_r = check_joint_limits(target_r, q_min, q_max)

        # 发送控制指令
        if phase == 0:
            server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
                                        [0.0]*7, [0.0]*7)
        else:
            print(f"\n[阶段{phase}] 期望内力 F_I_des={F_I_des:.2f}N, 实际内力 F_I_est={F_I_est:.2f}N, 滑动位移 δ={delta:.4f}, 摩擦裕度 fc={fc:.2f}")
            print(f"\n夹持目标姿态: 左臂 {target_l}, 右臂 {target_r}")

        # 记录
        row = np.concatenate([
            [exp_time, phase],
            cur_l_q, cur_r_q,
            cur_l_dq, cur_r_dq,
            F_l, F_r,
            [t_left, t_right],
            [F_E, F_I_des, F_I_est],
            [delta, delta_roll, fc, 1.0 if difr_active else 0.0],
            f_left_5, f_right_5
        ])
        data_file.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        record_count += 1

        # 打印
        if exp_time - last_print >= 0.1:
            phase_str = ["MOVE", "GRASP", "TEST"][phase]
            active_str = "DIFR-ON" if difr_active else "difr-off"
            fc_str = f"fc={fc:.2f}" + ("!" if fc > 1.0 else "")
            print(f"\r  t={exp_time:5.2f}s [{phase_str}] [{active_str}] | "
                  f"GMO={gmo_force:5.1f}N F_E={F_E:5.1f}N | "
                  f"F_I_des={F_I_des:5.1f} est={F_I_est:5.1f} | "
                  f"δ={delta:+.4f} {fc_str} | "
                  f"触觉 L={t_left:6.1f} R={t_right:6.1f}",
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
        print(f"\n[阶段2] 夹持物体，跟踪标称内力 F_0={args.F0}N ({args.grasp_time}s)...")
        grasp_start = time.time()
        while True:
            elapsed = time.time() - grasp_start
            exp_time = time.time() - exp_start
            if elapsed >= args.grasp_time:
                break
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段2] 夹持完成")

        # 阶段3: TEST
        print(f"\n[阶段3] 碰撞测试 ({args.test_time}s) — 可施加外部冲击，DIFR将激活")
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

        if tactile_process.is_alive():
            stop_event.set()
            tactile_process.join(timeout=5.0)
            if tactile_process.is_alive():
                tactile_process.terminate()
            print("[触觉] 子进程已停止")

        data_file.write(f"# 记录总帧数: {record_count}\n")
        data_file.close()
        print(f"[数据] 已保存 {record_count} 帧到: {output_path}")

        try:
            cur = server.manager.get_current_arm_states()
            server.manager.set_arm_poses(cur["left_q"], cur["right_q"], [0.0]*7, [0.0]*7)
            server.stop()
            print("[G1] 已停止")
        except Exception as e:
            print(f"[G1] 停止异常: {e}")

        pb.disconnect()
        print("[PyBullet] 已断开")
        print(f"\n[完成] 数据文件: {output_path}")


if __name__ == "__main__":
    main()

