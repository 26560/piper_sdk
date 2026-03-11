#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import math
import numpy as np
from piper_sdk import *

# =========================
# 配置
# =========================
CAN_NAME = "pipercan0"

INFER_JSON = "/home/swfu/ws/piper_ggcnn/ggcnn/captures/infer_result.json"
DEPTH_NPY = "/home/swfu/ws/piper_ggcnn/ggcnn/captures/depth_0000.npy"

# D405 内参
FX = 653.9111
FY = 653.9111
PPX = 645.5704
PPY = 363.8945

# 你的 depth_0000.npy 实测对应 1834 -> 0.1834 m，所以这里用 0.0001
DEPTH_SCALE_TO_M = 0.0001

# 目标点上方悬停高度，防止碰桌
HOVER_Z_OFFSET_M = 0.08

# # 持续发送时间
# SEND_SECONDS = 5.0
# DT = 0.01
# SPEED_PERCENT = 100

# =========================
# D405 -> 夹爪中心(ee) 的固定变换
# 如果悬停方向不对，只需要改这个矩阵
# =========================
T_EE_CAM = np.array([
    [0.0,  0.0,  1.0, -0.132],
    [-1.0, 0.0,  0.0,  0.000],
    [0.0, -1.0,  0.0,  0.046],
    [0.0,  0.0,  0.0,  1.000],
], dtype=np.float64)


# =========================
# 工具函数
# =========================
def rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def transform_point(T, p):
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T @ ph
    return out[:3]


def map_300_to_fullres(u_300, v_300, crop_x0, crop_y0, crop_side):
    scale = crop_side / 300.0
    u_img = crop_x0 + (u_300 + 0.5) * scale
    v_img = crop_y0 + (v_300 + 0.5) * scale
    return u_img, v_img


def robust_depth_at(depth_img, u, v, radius=3):
    h, w = depth_img.shape[:2]
    ui = int(round(u))
    vi = int(round(v))

    x1 = max(0, ui - radius)
    x2 = min(w, ui + radius + 1)
    y1 = max(0, vi - radius)
    y2 = min(h, vi + radius + 1)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]
    return float(np.median(valid))


def pixel_to_camera_xyz(u, v, z_raw):
    z = z_raw * DEPTH_SCALE_TO_M
    x = (u - PPX) * z / FX
    y = (v - PPY) * z / FY
    return np.array([x, y, z], dtype=np.float64)


def current_pose_from_piper(piper):
    msg = piper.GetArmEndPoseMsgs().end_pose

    # SDK demo里 EndPoseCtrl 输入是 mm*1000 / deg*1000
    # 所以反馈这里对应反解：
    x_m = msg.X_axis / 1000000.0
    y_m = msg.Y_axis / 1000000.0
    z_m = msg.Z_axis / 1000000.0

    rx_deg = msg.RX_axis / 1000.0
    ry_deg = msg.RY_axis / 1000.0
    rz_deg = msg.RZ_axis / 1000.0

    xyz_m = np.array([x_m, y_m, z_m], dtype=np.float64)
    rpy_deg = np.array([rx_deg, ry_deg, rz_deg], dtype=np.float64)
    return xyz_m, rpy_deg


def pose_to_command_units(xyz_m, rpy_deg):
    X = round(xyz_m[0] * 1000000.0)
    Y = round(xyz_m[1] * 1000000.0)
    Z = round(xyz_m[2] * 1000000.0)

    RX = round(rpy_deg[0] * 1000.0)
    RY = round(rpy_deg[1] * 1000.0)
    RZ = round(rpy_deg[2] * 1000.0)
    return X, Y, Z, RX, RY, RZ


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    # 1) 读取 GG-CNN 输出
    with open(INFER_JSON, "r", encoding="utf-8") as f:
        infer = json.load(f)

    depth = np.load(DEPTH_NPY).astype(np.float32)

    u_300 = infer["u_300"]
    v_300 = infer["v_300"]
    crop_x0 = infer["crop_x0"]
    crop_y0 = infer["crop_y0"]
    crop_side = infer["crop_side"]

    # 2) 300x300 抓取点 -> 原图像素
    u_img, v_img = map_300_to_fullres(
        u_300, v_300, crop_x0, crop_y0, crop_side
    )

    # 3) 取深度并反投影到相机坐标系
    z_raw = robust_depth_at(depth, u_img, v_img, radius=3)
    p_cam = pixel_to_camera_xyz(u_img, v_img, z_raw)

    # 4) 连接机械臂，按官方 demo 方式使能
    piper = C_PiperInterface_V2(CAN_NAME, False)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(1)

    # 5) 读取当前末端位姿
    cur_xyz_m, cur_rpy_deg = current_pose_from_piper(piper)

    # 6) 构建 T_base_ee
    R_base_ee = rpy_deg_to_matrix(cur_rpy_deg[0], cur_rpy_deg[1], cur_rpy_deg[2])
    T_base_ee = make_transform(R_base_ee, cur_xyz_m)

    # 7) 链式求解：base <- ee <- cam
    T_base_cam = T_base_ee @ T_EE_CAM
    target_xyz_m = transform_point(T_base_cam, p_cam)

    # 8) 在目标点上方悬停
    hover_xyz_m = target_xyz_m.copy()
    hover_xyz_m[2] += HOVER_Z_OFFSET_M

    # 9) 姿态保持当前值不变
    keep_rpy_deg = cur_rpy_deg.copy()

    # 10) 打印结果
    print("=" * 80)
    print("[INFO] infer_result:")
    print(json.dumps(infer, indent=2, ensure_ascii=False))
    print("[INFO] camera point (m):", p_cam.tolist())
    print("[INFO] u_img, v_img:", u_img, v_img)
    print("[INFO] depth_raw:", z_raw, " depth_m:", z_raw * DEPTH_SCALE_TO_M)
    print("[INFO] current end pose xyz(m):", cur_xyz_m.tolist())
    print("[INFO] current end pose rpy(deg):", keep_rpy_deg.tolist())
    print("[INFO] target point in base (m):", target_xyz_m.tolist())
    print("[INFO] hover point in base (m):", hover_xyz_m.tolist())
    print("=" * 80)

    # 11) 转成控制命令单位
    X, Y, Z, RX, RY, RZ = pose_to_command_units(hover_xyz_m, keep_rpy_deg)
    RX = -162606
    RY = 20572
    RZ = -48808
    

    # 12) 按官方 demo 风格持续发送 hover 位姿
    print(X, Y, Z, RX, RY, RZ, sep=',',)
    # piper.MotionCtrl_2(0x01, 0x00, SPEED_PERCENT, 0x00)
    # piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    print("[DONE] hover command finished")