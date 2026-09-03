#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-03 (v2: 三阶段 + QP数据)
Description: DIFR vs SIFR 双臂协作搬运实验数据可视化（论文级出图）

数据列格式（v2）:
  col 1:  time
  col 2:  phase (0=MOVE, 1=GRASP, 2=TEST)
  col 3-9:   left_q (7)
  col 10-16: right_q (7)
  col 17-23: left_dq (7)
  col 24-30: right_dq (7)
  col 31-33: left_gmo_F (3)
  col 34-36: right_gmo_F (3)
  col 37: tactile_left
  col 38: tactile_right
  col 39: F_E (估计外力)
  col 40: F_I_des (期望内力)
  col 41: F_I_est (实际内力)
  col 42: delta (滑动位移)
  col 43: delta_roll (肩roll偏移)
  col 44: difr_active
  col 45-49: left_tactile_5f (5)
  col 50-54: right_tactile_5f (5)

用法：
  python plot_experiment_results.py \
      --difr difr_experiment_xxx.txt \
      --sifr sifr_experiment_xxx.txt \
      --output-prefix comparison --format pdf
'''
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLOR_DIFR = '#1f77b4'
COLOR_SIFR = '#d62728'
COLOR_LEFT = '#2ca02c'
COLOR_RIGHT = '#ff7f0e'
PHASE_COLORS = ['#e8f4f8', '#fff3e0', '#fce4ec']  # MOVE, GRASP, TEST


def load_experiment_data(filepath):
    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        return None
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            try:
                row = np.array([float(x) for x in line.split()])
                data.append(row)
            except ValueError:
                continue
    if not data:
        print(f"[错误] 文件中无有效数据: {filepath}")
        return None
    D = np.array(data)
    print(f"[加载] {os.path.basename(filepath)}: {D.shape[0]} 帧, {D.shape[1]} 列")

    result = {
        'time': D[:, 0],
        'phase': D[:, 1],
        'left_q': D[:, 2:9],
        'right_q': D[:, 9:16],
        'left_dq': D[:, 16:23],
        'right_dq': D[:, 23:30],
        'left_gmo_F': D[:, 30:33],
        'right_gmo_F': D[:, 33:36],
        'tactile_left': D[:, 36],
        'tactile_right': D[:, 37],
        'F_E': D[:, 38],
        'F_I_des': D[:, 39],
        'F_I_est': D[:, 40],
        'delta': D[:, 41],
        'delta_roll': D[:, 42],
        'fc': D[:, 43],
        'difr_active': D[:, 44],
        'left_tactile_5f': D[:, 45:50],
        'right_tactile_5f': D[:, 50:55],
    }
    result['gmo_force_norm'] = (np.linalg.norm(result['left_gmo_F'], axis=1) +
                                  np.linalg.norm(result['right_gmo_F'], axis=1)) / 2.0
    result['tactile_sum'] = result['tactile_left'] + result['tactile_right']
    return result


def add_phase_shading(ax, data):
    '''添加三阶段背景色标注'''
    t = data['time']
    phase = data['phase']
    phase_names = ['MOVE', 'GRASP', 'TEST']
    for p in range(3):
        idx = np.where(phase == p)[0]
        if len(idx) > 0:
            t_start = t[idx[0]]
            t_end = t[idx[-1]]
            ax.axvspan(t_start, t_end, alpha=0.3, color=PHASE_COLORS[p], zorder=0)
            # 在顶部标注阶段名
            ylim = ax.get_ylim()
            ax.text((t_start + t_end) / 2, ylim[1] * 0.95, phase_names[p],
                    ha='center', va='top', fontsize=9, color='#555555',
                    style='italic', zorder=5)


def plot_gmo_force(difr, sifr, output_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    if difr:
        ax.plot(difr['time'], difr['gmo_force_norm'], color=COLOR_DIFR,
                label='DIFR (proposed)', linewidth=1.8)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['gmo_force_norm'], color=COLOR_SIFR,
                label='SIFR (baseline)', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Estimated external force $\|\mathbf{F}_{ext}\|$ [N]')
    ax.set_title('GMO-based Collision Force Estimation')
    ax.legend(loc='upper right')
    ax.grid(True, zorder=1)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_tactile_force(difr, sifr, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for idx, hand in enumerate(['left', 'right']):
        ax = axes[idx]
        if difr:
            ax.plot(difr['time'], difr[f'tactile_{hand}'], color=COLOR_DIFR,
                    label='DIFR', linewidth=1.5)
            add_phase_shading(ax, difr)
        if sifr:
            ax.plot(sifr['time'], sifr[f'tactile_{hand}'], color=COLOR_SIFR,
                    label='SIFR', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Tactile normal force $F_{tactile}$ [raw]')
        ax.set_title(f'{hand.capitalize()} Hand Tactile Contact Force')
        ax.legend()
        ax.grid(True, zorder=1)
        ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_internal_force(difr, sifr, output_path):
    '''核心图：期望内力 vs 实际内力（验证公式41）'''
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    if difr:
        ax = axes[0]
        ax.plot(difr['time'], difr['F_I_des'], color=COLOR_DIFR,
                label=r'$F_{int}^{des}$ (DIFR, Eq.41)', linewidth=2.0)
        ax.plot(difr['time'], difr['F_I_est'], color=COLOR_LEFT,
                label=r'$F_{int}^{est}$ (tactile)', linewidth=1.0, alpha=0.6)
        # 标注DIFR激活区域
        active_idx = np.where(difr['difr_active'] > 0.5)[0]
        if len(active_idx) > 0:
            for i in range(0, len(active_idx), max(1, len(active_idx)//20)):
                t_a = difr['time'][active_idx[i]]
                ax.axvline(x=t_a, color='red', alpha=0.05, linewidth=0.5)
        add_phase_shading(ax, difr)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Internal force $F_{int}$ [N]')
        ax.set_title('DIFR: Dynamic Internal Force (Eq. 41 QP)')
        ax.legend(loc='upper left')
        ax.grid(True, zorder=1)
        ax.set_xlim(left=0)

    if sifr:
        ax = axes[1]
        ax.plot(sifr['time'], sifr['F_I_des'], color=COLOR_SIFR,
                label=r'$F_{int}^{des}$ (SIFR, fixed)', linewidth=2.0)
        ax.plot(sifr['time'], sifr['F_I_est'], color=COLOR_LEFT,
                label=r'$F_{int}^{est}$ (tactile)', linewidth=1.0, alpha=0.6)
        add_phase_shading(ax, sifr)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Internal force $F_{int}$ [N]')
        ax.set_title('SIFR: Static Internal Force (fixed)')
        ax.legend(loc='upper left')
        ax.grid(True, zorder=1)
        ax.set_xlim(left=0)

    plt.suptitle(r'Comparison of Desired Internal Force Regulation (Eq. 41)',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_slip_delta(difr, sifr, output_path):
    '''滑动位移δ对比（DIFR抑制滑动 vs SIFR滑动累积）'''
    fig, ax = plt.subplots(figsize=(9, 4))
    if difr:
        ax.plot(difr['time'], difr['delta'] * 1000, color=COLOR_DIFR,
                label='DIFR (slip suppressed)', linewidth=1.8)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['delta'] * 1000, color=COLOR_SIFR,
                label='SIFR (slip accumulates)', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Slip displacement $\delta$ [mm]')
    ax.set_title('Slip Displacement: DIFR vs SIFR (Eq. 39-40)')
    ax.legend()
    ax.grid(True, zorder=1)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_friction_cone(difr, sifr, output_path):
    '''摩擦锥裕度fc对比（DIFR维持fc<1 vs SIFR可能fc>1滑动）'''
    fig, ax = plt.subplots(figsize=(9, 4))
    if difr:
        ax.plot(difr['time'], difr['fc'], color=COLOR_DIFR,
                label='DIFR (adaptive)', linewidth=1.8)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['fc'], color=COLOR_SIFR,
                label='SIFR (fixed)', linewidth=1.5, alpha=0.8)
    # fc=1临界线（超过即滑动）
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
               label=r'Friction cone limit $f_c=1$ (slip onset)')
    # fc=0.8预警线
    ax.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.0, alpha=0.5,
               label=r'DIFR activation threshold $f_c=0.8$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Friction cone margin $f_c = F_E / (\mu F_{int})$')
    ax.set_title('Friction Cone Margin: DIFR vs SIFR')
    ax.legend(loc='upper left')
    ax.grid(True, zorder=1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_delta_roll(difr, sifr, output_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    if difr:
        ax.plot(difr['time'], np.degrees(difr['delta_roll']), color=COLOR_DIFR,
                label='DIFR', linewidth=1.5)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], np.degrees(sifr['delta_roll']), color=COLOR_SIFR,
                label='SIFR', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Shoulder roll offset $\Delta\phi$ [deg]')
    ax.set_title('Internal Force Regulation Action (Shoulder Roll Offset)')
    ax.legend()
    ax.grid(True, zorder=1)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_summary(difr, sifr, output_path):
    '''2x2汇总对比图（论文主图）'''
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) GMO外力
    ax = axes[0, 0]
    if difr:
        ax.plot(difr['time'], difr['gmo_force_norm'], color=COLOR_DIFR, label='DIFR', linewidth=1.8)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['gmo_force_norm'], color=COLOR_SIFR, label='SIFR', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$\|\mathbf{F}_{ext}\|$ [N]')
    ax.set_title('(a) GMO Collision Force Estimation')
    ax.legend()
    ax.grid(True, zorder=1)

    # (b) 期望内力（核心）
    ax = axes[0, 1]
    if difr:
        ax.plot(difr['time'], difr['F_I_des'], color=COLOR_DIFR,
                label=r'DIFR $F_{int}^{des}$ (Eq.41)', linewidth=1.8)
    if sifr:
        ax.plot(sifr['time'], sifr['F_I_des'], color=COLOR_SIFR,
                label=r'SIFR $F_{int}^{des}$ (fixed)', linewidth=1.8, linestyle='--')
    if difr:
        add_phase_shading(ax, difr)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Desired internal force $F_{int}^{des}$ [N]')
    ax.set_title('(b) Desired Internal Force Regulation (Eq. 41)')
    ax.legend()
    ax.grid(True, zorder=1)

    # (c) 滑动位移
    ax = axes[1, 0]
    if difr:
        ax.plot(difr['time'], difr['delta'] * 1000, color=COLOR_DIFR, label='DIFR', linewidth=1.5)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['delta'] * 1000, color=COLOR_SIFR, label='SIFR', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Slip $\delta$ [mm]')
    ax.set_title('(c) Slip Displacement (Eq. 39-40)')
    ax.legend()
    ax.grid(True, zorder=1)

    # (d) 触觉力
    ax = axes[1, 1]
    if difr:
        ax.plot(difr['time'], difr['tactile_sum'], color=COLOR_DIFR, label='DIFR', linewidth=1.5)
        add_phase_shading(ax, difr)
    if sifr:
        ax.plot(sifr['time'], sifr['tactile_sum'], color=COLOR_SIFR, label='SIFR', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Tactile force $F_{tactile}$ [raw]')
    ax.set_title('(d) Tactile Contact Force (Both Hands)')
    ax.legend()
    ax.grid(True, zorder=1)

    plt.suptitle('DIFR vs SIFR: Dual-Arm Collaborative Manipulation Under External Disturbance',
                  fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def plot_joint_trajectories(data, method_name, output_path):
    if data is None:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Wrist Roll', 'Wrist Pitch', 'Wrist Yaw']
    for i in range(7):
        ax = axes[i // 4, i % 4]
        ax.plot(data['time'], np.degrees(data['left_q'][:, i]),
                color=COLOR_LEFT, label='Left', linewidth=1.2)
        ax.plot(data['time'], np.degrees(data['right_q'][:, i]),
                color=COLOR_RIGHT, label='Right', linewidth=1.2)
        add_phase_shading(ax, data)
        ax.set_title(joint_names[i], fontsize=11)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.legend(fontsize=9)
        ax.grid(True, zorder=1)
    ax = axes[1, 3]
    ax.plot(data['time'], data['gmo_force_norm'], color=COLOR_DIFR, linewidth=1.2)
    add_phase_shading(ax, data)
    ax.set_title('GMO External Force', fontsize=11)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.grid(True, zorder=1)
    plt.suptitle(f'{method_name} Joint Angle Trajectories', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[保存] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DIFR vs SIFR 实验数据可视化 (v2)")
    parser.add_argument('--difr', type=str, default=None)
    parser.add_argument('--sifr', type=str, default=None)
    parser.add_argument('--output-prefix', type=str, default='comparison')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'eps', 'svg'])
    args = parser.parse_args()

    if not args.difr and not args.sifr:
        print("[错误] 至少需要提供 --difr 或 --sifr 数据文件")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.output_prefix)
    ext = args.format

    difr_data = load_experiment_data(args.difr) if args.difr else None
    sifr_data = load_experiment_data(args.sifr) if args.sifr else None

    print("\n" + "=" * 60)
    print("  开始生成图表")
    print("=" * 60)

    plot_gmo_force(difr_data, sifr_data, f"{prefix}_fig1_gmo_force.{ext}")
    plot_tactile_force(difr_data, sifr_data, f"{prefix}_fig2_tactile_force.{ext}")
    plot_internal_force(difr_data, sifr_data, f"{prefix}_fig3_internal_force.{ext}")
    plot_slip_delta(difr_data, sifr_data, f"{prefix}_fig4_slip_delta.{ext}")
    plot_friction_cone(difr_data, sifr_data, f"{prefix}_fig5_friction_cone.{ext}")
    plot_delta_roll(difr_data, sifr_data, f"{prefix}_fig6_delta_roll.{ext}")
    plot_summary(difr_data, sifr_data, f"{prefix}_fig7_summary.{ext}")

    if difr_data:
        plot_joint_trajectories(difr_data, 'DIFR', f"{prefix}_fig8_difr_joints.{ext}")
    if sifr_data:
        plot_joint_trajectories(sifr_data, 'SIFR', f"{prefix}_fig9_sifr_joints.{ext}")

    print("\n" + "=" * 60)
    print(f"  所有图表已保存到: {args.output_dir}/")
    print(f"  文件名前缀: {args.output_prefix}_*.{ext}")
    print("=" * 60)

    # 统计信息
    if difr_data and sifr_data:
        print("\n[统计] TEST阶段关键指标对比:")
        for name, data in [('DIFR', difr_data), ('SIFR', sifr_data)]:
            test_mask = data['phase'] == 2
            if np.any(test_mask):
                print(f"  {name}:")
                print(f"    GMO力均值: {np.mean(data['gmo_force_norm'][test_mask]):.2f} N")
                print(f"    GMO力峰值: {np.max(data['gmo_force_norm'][test_mask]):.2f} N")
                print(f"    期望内力均值: {np.mean(data['F_I_des'][test_mask]):.2f} N")
                print(f"    期望内力峰值: {np.max(data['F_I_des'][test_mask]):.2f} N")
                print(f"    摩擦锥裕度峰值: {np.max(data['fc'][test_mask]):.2f} ({'滑动!' if np.max(data['fc'][test_mask]) > 1.0 else '安全'})")
                print(f"    滑动位移峰值: {np.max(np.abs(data['delta'][test_mask]))*1000:.2f} mm")
                print(f"    触觉力均值: {np.mean(data['tactile_sum'][test_mask]):.2f} raw")
                if name == 'DIFR':
                    print(f"    DIFR激活时间占比: {np.mean(data['difr_active'][test_mask])*100:.1f}%")


if __name__ == "__main__":
    main()

