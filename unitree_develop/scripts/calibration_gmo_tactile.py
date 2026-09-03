#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-03 (v5: 增加灵巧手自重重力补偿)
Description: 双臂GMO外力矩与触觉传感器标定对比实验

v5 新增：灵巧手自重重力补偿
  GMO只观测到手臂末端(wrist link)，不包含灵巧手。
  手的自重会在末端产生恒定重力，被GMO误检为"外力"。
  HandGravityCompensator 通过雅可比转置估计手自重的关节力矩，
  从GMO残差中抵消：r_compensated = r - J^T @ F_hand_gravity

功能：
  1. 双臂五次多项式运动到指定位置（10s），然后保持（10s），总时长20s
  2. 左右臂各自运行 GMO 估计关节外力矩
  3. 手自重重力补偿（抵消灵巧手自重产生的虚假外力）
  4. 通过雅可比转置将GMO关节力矩转换为末端力
  5. 读取灵巧手触觉传感器数据（normal_force1 法向，tangential_force1/2/3 切向）
  6. 实时对比显示 GMO末端力(补偿后) vs 触觉接触力
  7. 记录全过程数据到 txt
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
# 触觉传感器子进程（spawn 全新进程，隔离 DDS）
# ================================================================
def tactile_subscriber_worker(data_queue, log_queue, stop_event):
    try:
        import rclpy
        from rclpy.node import Node
        from ros2_stark_msgs.msg import TouchStatus
    except ImportError as e:
        log_queue.put(f"[触觉][子进程] 导入失败: {e}")
        return

    FINGER_COUNT = 5

    class TactileNode(Node):
        def __init__(self):
            super().__init__("calibration_tactile_sub")
            self.left_sub = self.create_subscription(
                TouchStatus, "/left_hand/touch_status_126", self.left_cb, 10)
            self.right_sub = self.create_subscription(
                TouchStatus, "/right_hand/touch_status_127", self.right_cb, 10)
            log_queue.put("[触觉][子进程] 订阅已创建")

        def _extract(self, msg):
            nf1 = np.zeros(FINGER_COUNT)
            nf2 = np.zeros(FINGER_COUNT)
            nf3 = np.zeros(FINGER_COUNT)
            tf1 = np.zeros(FINGER_COUNT)
            tf2 = np.zeros(FINGER_COUNT)
            tf3 = np.zeros(FINGER_COUNT)
            for i in range(min(FINGER_COUNT, len(msg.data))):
                fd = msg.data[i]
                nf1[i] = float(fd.normal_force1)
                nf2[i] = float(fd.normal_force2)
                nf3[i] = float(fd.normal_force3)
                tf1[i] = float(fd.tangential_force1)
                tf2[i] = float(fd.tangential_force2)
                tf3[i] = float(fd.tangential_force3)
            return nf1, nf2, nf3, tf1, tf2, tf3

        def left_cb(self, msg):
            nf1, nf2, nf3, tf1, tf2, tf3 = self._extract(msg)
            data_queue.put(("left", nf1.tolist(), nf2.tolist(), nf3.tolist(),
                            tf1.tolist(), tf2.tolist(), tf3.tolist(), time.time()))

        def right_cb(self, msg):
            nf1, nf2, nf3, tf1, tf2, tf3 = self._extract(msg)
            data_queue.put(("right", nf1.tolist(), nf2.tolist(), nf3.tolist(),
                            tf1.tolist(), tf2.tolist(), tf3.tolist(), time.time()))

    try:
        rclpy.init()
        node = TactileNode()
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()
        log_queue.put("[触觉][子进程] 已安全退出")
    except Exception as e:
        log_queue.put(f"[触觉][子进程] 错误: {e}")
        import traceback
        log_queue.put(traceback.format_exc())


# ================================================================
# 触觉数据容器
# ================================================================
class TactileData:
    def __init__(self):
        self.left_nf1 = np.zeros(5)
        self.left_nf2 = np.zeros(5)
        self.left_nf3 = np.zeros(5)
        self.right_nf1 = np.zeros(5)
        self.right_nf2 = np.zeros(5)
        self.right_nf3 = np.zeros(5)
        self.left_tf1 = np.zeros(5)
        self.left_tf2 = np.zeros(5)
        self.left_tf3 = np.zeros(5)
        self.right_tf1 = np.zeros(5)
        self.right_tf2 = np.zeros(5)
        self.right_tf3 = np.zeros(5)
        self.left_received = False
        self.right_received = False
        self._lock = threading.Lock()

    def update(self, hand, nf1, nf2, nf3, tf1, tf2, tf3):
        with self._lock:
            if hand == "left":
                self.left_nf1 = np.array(nf1, dtype=float)
                self.left_nf2 = np.array(nf2, dtype=float)
                self.left_nf3 = np.array(nf3, dtype=float)
                self.left_tf1 = np.array(tf1, dtype=float)
                self.left_tf2 = np.array(tf2, dtype=float)
                self.left_tf3 = np.array(tf3, dtype=float)
                self.left_received = True
            else:
                self.right_nf1 = np.array(nf1, dtype=float)
                self.right_nf2 = np.array(nf2, dtype=float)
                self.right_nf3 = np.array(nf3, dtype=float)
                self.right_tf1 = np.array(tf1, dtype=float)
                self.right_tf2 = np.array(tf2, dtype=float)
                self.right_tf3 = np.array(tf3, dtype=float)
                self.right_received = True

    def get(self):
        with self._lock:
            return (self.left_nf1.copy(), self.left_nf2.copy(), self.left_nf3.copy(),
                    self.right_nf1.copy(), self.right_nf2.copy(), self.right_nf3.copy(),
                    self.left_tf1.copy(), self.left_tf2.copy(), self.left_tf3.copy(),
                    self.right_tf1.copy(), self.right_tf2.copy(), self.right_tf3.copy(),
                    self.left_received, self.right_received)


# ================================================================
# GMO 广义动量观测器
# ================================================================
class GMOObserver:
    def __init__(self, n_dof=7, gain=15.0, internal_gain=0.5):
        self.n_dof = n_dof
        self.K = gain * np.eye(n_dof)
        # 积分项
        self.Ki = internal_gain * np.eye(n_dof)  # 积分增益
        self.integral = np.zeros(n_dof)            # 积分项

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
        # 比例残差
        r_p = self.K @ (P - self.P_hat)
        # 积分项（带抗饱和限幅）
        self.integral += r_p * dt
        self.integral = np.clip(self.integral, -20.0, 20.0)  # 抗饱和
        # 最终残差 = 比例 + 积分
        self.r = r_p + self.Ki @ self.integral
        return self.r.copy()


# ================================================================
# 灵巧手自重重力补偿器
# ================================================================
class HandGravityCompensator:
    '''
    估计灵巧手自重对关节力矩的贡献，从GMO残差中抵消。

    原理：
      GMO观测到手臂末端(wrist link)，但灵巧手安装在末端上，
      手的重力 F_hand = [0, 0, -m*g] 作用在末端，
      通过雅可比转置映射为关节力矩：τ_hand = J_p^T @ F_hand
      补偿：r_compensated = r_gmo - τ_hand

    简单版本假设：
      - 手的质心在末端link原点（忽略质心偏移产生的力矩）
      - 只补偿重力的力分量，不补偿手的转动惯量动态效应
    '''
    def __init__(self, hand_mass=0.45, g=9.81):
        '''
        Args:
            hand_mass: 灵巧手质量 (kg)。Revo2约0.4-0.5kg，含手指约0.45kg
            g: 重力加速度 (m/s^2)
        '''
        self.hand_mass = hand_mass
        self.g = g
        # 手重力在世界坐标系下（始终竖直向下）
        self.F_hand_world = np.array([0.0, 0.0, -hand_mass * g])
        print(f"[手补偿] 手质量: {hand_mass}kg, 重力: {hand_mass*g:.2f}N "
              f"(方向: [0,0,-{hand_mass*g:.2f}])")

    def compute_hand_torque(self, J_p):
        '''
        计算手自重在关节空间产生的力矩
        Args:
            J_p: 末端位置雅可比 (3x7)，世界坐标系
        Returns:
            tau_hand: 手自重产生的关节力矩 (7,)
        '''
        return J_p.T @ self.F_hand_world

    def compute_hand_ee_force(self):
        '''手自重对应的末端力（世界坐标系，3维）'''
        return self.F_hand_world.copy()

    def compensate(self, r_gmo, J_p):
        '''
        从GMO残差中抵消手自重
        Args:
            r_gmo: GMO原始估计的关节外力矩 (7,)
            J_p: 末端位置雅可比 (3x7)
        Returns:
            r_compensated: 补偿后的关节外力矩 (7,)
            tau_hand: 被抵消的手自重力矩 (7,)
        '''
        tau_hand = self.compute_hand_torque(J_p)
        r_compensated = r_gmo - tau_hand
        return r_compensated, tau_hand


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
        print(f"[PyBullet] 总旋转关节数: {self.n_total}")

        self.joint_to_pos = {jid: idx for idx, jid in enumerate(self.all_revolute)}

        # 按关节名称匹配左右臂
        self.left_joints = []
        self.right_joints = []
        for jid in self.all_revolute:
            name = p.getJointInfo(self.robot_id, jid)[1].decode().lower()
            if name.startswith("left_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.left_joints.append(jid)
            elif name.startswith("right_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.right_joints.append(jid)

        if len(self.left_joints) != 7:
            print(f"[PyBullet] 警告: 左臂名称匹配得到 {len(self.left_joints)} 个，回退前7个")
            self.left_joints = self.all_revolute[:7]
        if len(self.right_joints) != 7:
            print(f"[PyBullet] 警告: 右臂名称匹配得到 {len(self.right_joints)} 个，回退后7个")
            self.right_joints = self.all_revolute[-7:]

        self.left_ee = self.left_joints[-1] + 1
        self.right_ee = self.right_joints[-1] + 1
        self.left_pos = [self.joint_to_pos[jid] for jid in self.left_joints]
        self.right_pos = [self.joint_to_pos[jid] for jid in self.right_joints]

        print(f"[PyBullet] 左臂: {self.left_joints}, EE: {self.left_ee}")
        print(f"[PyBullet] 右臂: {self.right_joints}, EE: {self.right_ee}")

    def _build_full_vector(self, arm, q7, dq7=None):
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
        full_q, full_dq = self._build_full_vector(arm, q, dq)
        zero = [0.0] * self.n_total
        pos = self.left_pos if arm == "left" else self.right_pos

        G_full = np.array(p.calculateInverseDynamics(
            self.robot_id, full_q.tolist(), zero, zero))
        Cq_full = np.array(p.calculateInverseDynamics(
            self.robot_id, full_q.tolist(), full_dq.tolist(), zero)) - G_full

        G = G_full[pos]
        Cq = Cq_full[pos]

        M = []
        for k in range(7):
            aa = np.zeros(self.n_total)
            aa[pos[k]] = 1.0
            T_full = np.array(p.calculateInverseDynamics(
                self.robot_id, full_q.tolist(), zero, aa.tolist())) - G_full
            M.append(T_full[pos])
        M = np.array(M).T
        return M, Cq, G

    def compute_jacobian(self, arm, q, dq):
        full_q, full_dq = self._build_full_vector(arm, q, dq)
        ee = self.left_ee if arm == "left" else self.right_ee
        pos = self.left_pos if arm == "left" else self.right_pos

        J_p, _ = p.calculateJacobian(
            self.robot_id, ee, [0.0, 0.0, 0.0],
            full_q.tolist(), full_dq.tolist(), [0.0] * self.n_total)
        J_p = np.array(J_p)[:, pos]
        return J_p

    def disconnect(self):
        p.disconnect(self.client)


# ================================================================
# 工具函数
# ================================================================
def quintic_interpolate(start, goal, t, duration):
    ratio = np.clip(t / duration, 0.0, 1.0)
    s = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
    return start + (goal - start) * s


def joint_torque_to_ee_force(r, J):
    '''r = J^T @ F  →  F = pinv(J^T) @ r'''
    return np.linalg.pinv(J.T) @ r


def drain_queue(queue):
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except Exception:
            break
    return items


def print_status(exp_time, phase, F_left_raw, F_right_raw,
                  F_left_comp, F_right_comp,
                  F_hand_left, F_hand_right,
                  l_n_sum, r_n_sum, l_t_sum, r_t_sum):
    phase_str = "MOVE" if phase == 0 else "HOLD"
    print(f"\r  t={exp_time:5.2f}s [{phase_str}] | "
          f"GMO末端力(补偿后): L={np.linalg.norm(F_left_comp):6.1f} R={np.linalg.norm(F_right_comp):6.1f} N | "
          f"手自重补偿: L={np.linalg.norm(F_hand_left):5.1f} R={np.linalg.norm(F_hand_right):5.1f} N | "
          f"触觉法向: L={l_n_sum:7.1f} R={r_n_sum:7.1f}",
          end="", flush=True)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="双臂GMO外力矩与触觉传感器标定对比实验(含手自重补偿)")
    parser.add_argument('--left_q', type=float, nargs=7, required=True,
                        metavar=('q1','q2','q3','q4','q5','q6','q7'))
    parser.add_argument('--right_q', type=float, nargs=7, required=True,
                        metavar=('q1','q2','q3','q4','q5','q6','q7'))
    parser.add_argument('--move-time', type=float, default=10.0)
    parser.add_argument('--hold-time', type=float, default=10.0)
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--gmo-gain', type=float, default=5.0)
    parser.add_argument('--hand-mass', type=float, default=0.45,
                        help='灵巧手质量(kg)，用于自重补偿，默认0.45kg')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--urdf', type=str,
                        default='description/g1_14dof_brainco_hand.urdf')
    parser.add_argument('--no-tactile', action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print("  双臂 GMO ↔ 触觉 标定实验 (v5: 含灵巧手自重补偿)")
    print("=" * 80)
    print(f"  左臂目标:  {np.array(args.left_q)}")
    print(f"  右臂目标:  {np.array(args.right_q)}")
    print(f"  运动/保持: {args.move_time}s / {args.hold_time}s")
    print(f"  GMO增益:   {args.gmo_gain}")
    print(f"  手质量:    {args.hand_mass}kg (自重补偿)")
    print(f"  网卡:      {args.interface}")
    print("=" * 80)

    # 输出文件
    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"calibration_gmo_tactile_{ts}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    # ---- PyBullet ----
    print("\n[初始化] 加载 PyBullet 双臂模型...")
    pb = DualArmPyBullet(args.urdf)

    # ---- GMO ----
    gmo_left = GMOObserver(n_dof=7, gain=args.gmo_gain)
    gmo_right = GMOObserver(n_dof=7, gain=args.gmo_gain)

    # ---- 手自重补偿器（左右臂各一个）----
    hand_comp_left = HandGravityCompensator(hand_mass=args.hand_mass)
    hand_comp_right = HandGravityCompensator(hand_mass=args.hand_mass)

    # ---- G1 ----
    print(f"\n[初始化] 连接 G1 (网卡: {args.interface})...")
    server = G1Server(network_interface=args.interface,
                      shm_name="dev/shm/6_axis_force_shm")
    server.start()
    time.sleep(0.5)

    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)

    # ---- 触觉子进程 ----
    tactile_data = TactileData()
    tactile_process = None
    data_queue = None
    log_queue = None
    stop_event = None

    if not args.no_tactile:
        print("\n[触觉] 启动触觉订阅子进程 (spawn)...")
        ctx = mp.get_context('spawn')
        data_queue = ctx.Queue(maxsize=200)
        log_queue = ctx.Queue(maxsize=100)
        stop_event = ctx.Event()
        tactile_process = ctx.Process(
            target=tactile_subscriber_worker,
            args=(data_queue, log_queue, stop_event),
            daemon=True)
        tactile_process.start()
        print(f"[触觉] 子进程 PID: {tactile_process.pid}")
        time.sleep(4.0)
        for msg in drain_queue(log_queue):
            print(f"  {msg}")
    else:
        print("\n[触觉] 已禁用")

    # ---- 数据文件 ----
    data_file = open(output_path, 'w')
    data_file.write("# 双臂GMO外力矩与触觉传感器标定对比实验 (v5: 含手自重补偿)\n")
    data_file.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    data_file.write(f"# 左臂目标: {goal_l.tolist()}\n")
    data_file.write(f"# 右臂目标: {goal_r.tolist()}\n")
    data_file.write(f"# 手质量: {args.hand_mass}kg\n")
    data_file.write("# 列说明:\n")
    data_file.write("#   1: time, 2: phase(0=move,1=hold)\n")
    data_file.write("#   3-9: left_q, 10-16: right_q\n")
    data_file.write("#   17-23: left_r_raw (GMO原始残差, Nm)\n")
    data_file.write("#   24-30: right_r_raw\n")
    data_file.write("#   31-37: left_r_comp (补偿手自重后残差, Nm)\n")
    data_file.write("#   38-44: right_r_comp\n")
    data_file.write("#   45-47: left_F_raw (GMO原始末端力, N)\n")
    data_file.write("#   48-50: right_F_raw\n")
    data_file.write("#   51-53: left_F_comp (补偿后末端力, N)\n")
    data_file.write("#   54-56: right_F_comp\n")
    data_file.write("#   57-59: left_F_hand (手自重末端力, N)\n")
    data_file.write("#   60-62: right_F_hand\n")
    data_file.write("#   63-67: left_nf1 (5指法向力), 68-72: right_nf1\n")
    data_file.write("#   73-77: left_tangential(5指), 78-82: right_tangential\n")
    data_file.write("#   83: left_n_sum, 84: right_n_sum, 85: left_t_sum, 86: right_t_sum\n")
    data_file.write("#" + "=" * 80 + "\n")

    # ============================================================
    # 主循环
    # ============================================================
    print("\n" + "=" * 80)
    print("  实验开始 — 保持阶段可施加外力标定")
    print("=" * 80)

    t_step = 0.01
    exp_start = time.time()
    last_print = 0.0
    record_count = 0
    last_time = time.time()
    gmo_left.reset()
    gmo_right.reset()

    def run_one_step(exp_time, phase):
        nonlocal last_time, last_print, record_count

        now = time.time()
        dt = now - last_time
        last_time = now

        # 触觉
        if data_queue:
            for item in drain_queue(data_queue):
                hand, nf1, nf2, nf3, tf1, tf2, tf3, ts = item
                tactile_data.update(hand, nf1, nf2, nf3, tf1, tf2, tf3)
        if log_queue:
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

        pb.update_states(cur_l_q, cur_r_q)
        # print(f"\n  t={exp_time:5.2f}s [{phase}] | 左臂q: {cur_l_q} | 右臂q: {cur_r_q}", end="", flush=True)

        # ===== 左臂 GMO + 手自重补偿 =====
        M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
        r_l_raw = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)
        J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
        # 手自重补偿
        r_l_comp, tau_hand_l = hand_comp_left.compensate(r_l_raw, J_l)
        F_hand_l = hand_comp_left.compute_hand_ee_force()
        # 末端力（补偿前后）
        F_l_raw = joint_torque_to_ee_force(r_l_raw, J_l)
        F_l_comp = joint_torque_to_ee_force(r_l_comp, J_l)

        # ===== 右臂 GMO + 手自重补偿 =====
        M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
        r_r_raw = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
        J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
        r_r_comp, tau_hand_r = hand_comp_right.compensate(r_r_raw, J_r)
        F_hand_r = hand_comp_right.compute_hand_ee_force()
        F_r_raw = joint_torque_to_ee_force(r_r_raw, J_r)
        F_r_comp = joint_torque_to_ee_force(r_r_comp, J_r)

        # 触觉
        (lnf1, lnf2, lnf3, rnf1, rnf2, rnf3,
         ltf1, ltf2, ltf3, rtf1, rtf2, rtf3, _, _) = tactile_data.get()
        l_n_sum = float(np.sum(lnf1))
        r_n_sum = float(np.sum(rnf1))
        l_t_per = np.sqrt(ltf1**2 + ltf2**2 + ltf3**2)
        r_t_per = np.sqrt(rtf1**2 + rtf2**2 + rtf3**2)
        l_t_sum = float(np.sum(l_t_per))
        r_t_sum = float(np.sum(r_t_per))

        # 记录
        row = np.concatenate([
            [exp_time, phase],
            cur_l_q, cur_r_q,
            r_l_raw, r_r_raw,        # GMO原始残差
            r_l_comp, r_r_comp,      # 补偿后残差
            F_l_raw, F_r_raw,        # 原始末端力
            F_l_comp, F_r_comp,      # 补偿后末端力
            F_hand_l, F_hand_r,      # 手自重末端力
            lnf1, rnf1,               # 5指法向力
            l_t_per, r_t_per,         # 5指切向力
            [l_n_sum, r_n_sum, l_t_sum, r_t_sum]
        ])
        data_file.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        record_count += 1

        # 打印
        if exp_time - last_print >= 0.1:
            print_status(exp_time, phase, F_l_raw, F_r_raw,
                         F_l_comp, F_r_comp,
                         F_hand_l, F_hand_r,
                         l_n_sum, r_n_sum, l_t_sum, r_t_sum)
            last_print = exp_time

    try:
        # 阶段1: 运动
        print(f"\n[阶段1] 运动到目标位置 ({args.move_time}s)...")
        move_start = time.time()
        while True:
            elapsed = time.time() - move_start
            exp_time = time.time() - exp_start
            if elapsed >= args.move_time:
                server.manager.set_arm_poses(goal_l.tolist(), goal_r.tolist(),
                                               [0.0]*7, [0.0]*7)
                break
            target_l = quintic_interpolate(start_l, goal_l, elapsed, args.move_time)
            target_r = quintic_interpolate(start_r, goal_r, elapsed, args.move_time)
            # print(f"\n  t={exp_time:5.2f}s [MOVE] | 左臂目标: {target_l} | 右臂目标: {target_r}\n", end="", flush=True)
            server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
                                           [0.0]*7, [0.0]*7)
            run_one_step(exp_time, 0)
            time.sleep(t_step)
        print(f"\n[阶段1] 到达目标位置")

        # 阶段2: 保持
        print(f"\n[阶段2] 保持位置 ({args.hold_time}s)...")
        hold_start = time.time()
        while True:
            elapsed = time.time() - hold_start
            exp_time = time.time() - exp_start
            # if elapsed >= args.hold_time:
            #     break
            server.manager.set_arm_poses(goal_l.tolist(), goal_r.tolist(),
                                           [0.0]*7, [0.0]*7)
            run_one_step(exp_time, 1)
            time.sleep(t_step)
        print(f"\n[阶段2] 保持完成")

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

        if tactile_process and tactile_process.is_alive():
            stop_event.set()
            tactile_process.join(timeout=5.0)
            if tactile_process.is_alive():
                tactile_process.terminate()
            for msg in drain_queue(log_queue):
                print(f"  [触觉] {msg}")
            print("[触觉] 子进程已停止")

        data_file.write(f"# 记录总帧数: {record_count}\n")
        data_file.close()
        print(f"[数据] 已保存 {record_count} 帧到: {output_path}")

        try:
            cur = server.manager.get_current_arm_states()
            server.manager.set_arm_poses(cur["left_q"], cur["right_q"],
                                           [0.0]*7, [0.0]*7)
            server.stop()
            print("[G1] 已停止")
        except Exception as e:
            print(f"[G1] 停止异常: {e}")

        pb.disconnect()
        print("[PyBullet] 已断开")
        print(f"\n[完成] 数据文件: {output_path}")


if __name__ == "__main__":
    main()

