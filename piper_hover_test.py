#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import socket
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from piper_sdk import *


SOCKET_PATH = "/tmp/piper_hover.sock"
CAN_NAME = "pipercan0"
GRIPPER_SOCKET = "/tmp/pika_gripper.sock"
PHOTO_SOCKET = "/tmp/piper_take_photo.sock"

INFER_JSON = Path("/home/swfu/ws/piper_ggcnn/ggcnn/captures/infer_result.json")
DEPTH_NPY = Path("/home/swfu/ws/piper_ggcnn/ggcnn/captures/depth_0000.npy")
CAPTURE_POSE_JSON = Path("/home/swfu/ws/piper_ggcnn/ggcnn/captures/pose_0000.json")
RELEASE_DEBUG_PNG = Path("/home/swfu/ws/piper_ggcnn/ggcnn/captures/release_depth_debug.png")
HAND_EYE_JSON = Path("/home/swfu/ws/handeye_out/handeye_best.json")
FIXED_CAMERA_INFO_JSON = Path(
    os.environ.get(
        "HAND_EYE_CAMERA_INFO_JSON",
        "/home/swfu/ws/piper_ggcnn/ggcnn/captures/camera_info.json",
    )
).expanduser()

DEPTH_SCALE_TO_M = 0.0001
HOVER_Z_OFFSET_M = -0.028

APPROACH_WAIT_SEC = 1.5
CLOSE_WAIT_SEC = 0.5
RELEASE_WAIT_SEC = 2.0

Z_PROTECT = 171500
RELEASE_Z_CLEARANCE_M = 0.015
RELEASE_DEPTH_EPSILON_RAW = 80.0
RELEASE_IQR_MARGIN_RAW = 200.0
RELEASE_SURFACE_PERCENTILE = 10.0

# 预放置位置
DEFAULT_PRE_RELEASE_POSE_CMD = [221813, 128064, 252747, 180000, 17678, -150000]
LEVEL_PRE_RELEASE_POSE_CMD = {
    "default": DEFAULT_PRE_RELEASE_POSE_CMD,
    "1": [-200409, 158528, 251952, 179844, 25004, -37527],
    "2": [262112, 95401, 301972, 180000, 27678, -160000],
    "3": [274696, 48436, 301972, 180000, 27678, -170000],
    "4": [278933, 0, 301972, 180000, 27678, 180000],
}

# 正式放置位置
DEFAULT_FORMAL_RELEASE_POSE_CMD = list(DEFAULT_PRE_RELEASE_POSE_CMD)
LEVEL_FORMAL_RELEASE_POSE_CMD = {
    "default": list(DEFAULT_FORMAL_RELEASE_POSE_CMD),
    "1": [-322236, 255206, 189165, 179901, 20073, -37539],
    "2": list(LEVEL_PRE_RELEASE_POSE_CMD["2"]),
    "3": list(LEVEL_PRE_RELEASE_POSE_CMD["3"]),
    "4": list(LEVEL_PRE_RELEASE_POSE_CMD["4"]),
}

# 自适应高度ROI
LEVEL_RELEASE_ROI = {
    "default": (784, 291, 984, 541),
    "1": (580, 70, 816, 336),
    "2": (333, 245, 640, 418),
    "3": (640, 245, 947, 418),
    "4": (784, 291, 984, 541),
}

T_FLANGE_TCP = np.eye(4, dtype=np.float64)
T_FLANGE_TCP[:3, 3] = np.array([0.0019368, 0.0011535, 0.1921555], dtype=np.float64)

with open(HAND_EYE_JSON, "r", encoding="utf-8") as file_obj:
    HANDEYE = json.load(file_obj)
T_FLANGE_CAM = np.array(HANDEYE["T_cam2gripper"], dtype=np.float64)
T_TCP_CAM = np.linalg.inv(T_FLANGE_TCP) @ T_FLANGE_CAM

with open(FIXED_CAMERA_INFO_JSON, "r", encoding="utf-8") as file_obj:
    CAMERA_INFO = json.load(file_obj)

if all(key in CAMERA_INFO for key in ["depth_fx", "depth_fy", "depth_ppx", "depth_ppy"]):
    FX = float(CAMERA_INFO["depth_fx"])
    FY = float(CAMERA_INFO["depth_fy"])
    PPX = float(CAMERA_INFO["depth_ppx"])
    PPY = float(CAMERA_INFO["depth_ppy"])
elif all(key in CAMERA_INFO for key in ["fx", "fy", "ppx", "ppy"]):
    FX = float(CAMERA_INFO["fx"])
    FY = float(CAMERA_INFO["fy"])
    PPX = float(CAMERA_INFO["ppx"])
    PPY = float(CAMERA_INFO["ppy"])
else:
    raise RuntimeError(f"unsupported camera_info format: {FIXED_CAMERA_INFO_JSON}")


def recv_one_line(conn: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode("utf-8").strip()


def send_one_line(conn: socket.socket, obj: dict):
    msg = json.dumps(obj, ensure_ascii=False) + "\n"
    conn.sendall(msg.encode("utf-8"))


def rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def transform_point(transform, point):
    point_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return (transform @ point_h)[:3]


def map_input_to_fullres(u_300, v_300, crop_x0, crop_y0, crop_side, input_side=300):
    scale = crop_side / float(input_side)
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
    valid = patch[(patch > 0) & (patch < 65000)]
    if valid.size == 0:
        raise ValueError(f"No valid depth around pixel ({u:.2f}, {v:.2f})")
    return float(np.median(valid))


def pixel_to_camera_xyz(u, v, z_raw):
    z = z_raw * DEPTH_SCALE_TO_M
    x = (u - PPX) * z / FX
    y = (v - PPY) * z / FY
    return np.array([x, y, z], dtype=np.float64)


def pose_json_to_xyz_rpy_deg(pose):
    xyz_m = np.array([pose["x_m"], pose["y_m"], pose["z_m"]], dtype=np.float64)
    raw = pose.get("raw", {})
    if {"rx", "ry", "rz"} <= raw.keys():
        rpy_deg = np.array(
            [raw["rx"] / 1000.0, raw["ry"] / 1000.0, raw["rz"] / 1000.0],
            dtype=np.float64,
        )
    else:
        rpy_deg = np.array(
            [
                math.degrees(pose["rx_rad"]),
                math.degrees(pose["ry_rad"]),
                math.degrees(pose["rz_rad"]),
            ],
            dtype=np.float64,
        )
    return xyz_m, rpy_deg


def pose_to_command_units(xyz_m, rpy_deg):
    return (
        round(xyz_m[0] * 1000000.0),
        round(xyz_m[1] * 1000000.0),
        round(xyz_m[2] * 1000000.0),
        round(rpy_deg[0] * 1000.0),
        round(rpy_deg[1] * 1000.0),
        round(rpy_deg[2] * 1000.0),
    )


def current_pose_from_piper(piper):
    msg = piper.GetArmEndPoseMsgs().end_pose
    xyz_m = np.array(
        [
            msg.X_axis / 1000000.0,
            msg.Y_axis / 1000000.0,
            msg.Z_axis / 1000000.0,
        ],
        dtype=np.float64,
    )
    rpy_deg = np.array(
        [
            msg.RX_axis / 1000.0,
            msg.RY_axis / 1000.0,
            msg.RZ_axis / 1000.0,
        ],
        dtype=np.float64,
    )
    return xyz_m, rpy_deg


def flange_to_tcp_xyz(flange_xyz_m, flange_rpy_deg):
    rotation = rpy_deg_to_matrix(flange_rpy_deg[0], flange_rpy_deg[1], flange_rpy_deg[2])
    return flange_xyz_m + rotation @ T_FLANGE_TCP[:3, 3]


def tcp_target_to_flange_cmd(target_tcp_xyz_m, cmd_rpy_deg):
    rotation = rpy_deg_to_matrix(cmd_rpy_deg[0], cmd_rpy_deg[1], cmd_rpy_deg[2])
    return target_tcp_xyz_m - rotation @ T_FLANGE_TCP[:3, 3]


def joint_cmd_to_pose_cmd(joint_cmd):
    fk_solver = C_PiperForwardKinematics()
    joint_rad = [math.radians(value / 1000.0) for value in joint_cmd]
    pose_mm_deg = fk_solver.CalFK(joint_rad)[-1]
    xyz_m = np.array(pose_mm_deg[:3], dtype=np.float64) / 1000.0
    rpy_deg = np.array(pose_mm_deg[3:], dtype=np.float64)
    return list(pose_to_command_units(xyz_m, rpy_deg))


def resolve_pre_release_pose_cmd(level, pose_override=None, joint_override=None):
    if pose_override is not None:
        if not isinstance(pose_override, (list, tuple)) or len(pose_override) != 6:
            raise ValueError("pre_release_pose_cmd must contain 6 values")
        return [int(value) for value in pose_override]

    if joint_override is not None:
        if not isinstance(joint_override, (list, tuple)) or len(joint_override) != 6:
            raise ValueError("release_joint_cmd must contain 6 values")
        return joint_cmd_to_pose_cmd([int(value) for value in joint_override])

    level_key = str(level).strip().upper()
    if level_key in LEVEL_PRE_RELEASE_POSE_CMD:
        return list(LEVEL_PRE_RELEASE_POSE_CMD[level_key])
    return list(LEVEL_PRE_RELEASE_POSE_CMD["default"])


def resolve_formal_release_pose_cmd(level, pose_override=None, joint_override=None):
    if pose_override is not None:
        if not isinstance(pose_override, (list, tuple)) or len(pose_override) != 6:
            raise ValueError("formal_release_pose_cmd must contain 6 values")
        return [int(value) for value in pose_override]

    if joint_override is not None:
        if not isinstance(joint_override, (list, tuple)) or len(joint_override) != 6:
            raise ValueError("formal_release_joint_cmd must contain 6 values")
        return joint_cmd_to_pose_cmd([int(value) for value in joint_override])

    level_key = str(level).strip().upper()
    if level_key in LEVEL_FORMAL_RELEASE_POSE_CMD:
        return list(LEVEL_FORMAL_RELEASE_POSE_CMD[level_key])
    return list(LEVEL_FORMAL_RELEASE_POSE_CMD["default"])


def pose_cmd_to_xyz_rpy_deg(pose_cmd):
    if not isinstance(pose_cmd, (list, tuple)) or len(pose_cmd) != 6:
        raise ValueError("pose_cmd must contain 6 values")

    pose_cmd = [int(value) for value in pose_cmd]
    xyz_m = np.array(pose_cmd[:3], dtype=np.float64) / 1000000.0
    rpy_deg = np.array(pose_cmd[3:], dtype=np.float64) / 1000.0
    return xyz_m, rpy_deg


def resolve_release_roi(level, image_shape):
    h, w = image_shape[:2]
    level_key = str(level).strip().upper()
    x0_px, y0_px, x1_px, y1_px = LEVEL_RELEASE_ROI.get(
        level_key, LEVEL_RELEASE_ROI["default"]
    )
    x0 = max(0, min(w - 1, int(x0_px)))
    y0 = max(0, min(h - 1, int(y0_px)))
    x1 = max(x0 + 1, min(w, int(x1_px)))
    y1 = max(y0 + 1, min(h, int(y1_px)))
    return x0, y0, x1, y1


def filtered_surface_depth_in_roi(depth_img, roi, valid_max=65000):
    x0, y0, x1, y1 = roi
    patch = depth_img[y0:y1, x0:x1]
    valid_mask = (patch > 0) & (patch < valid_max)
    valid = patch[valid_mask].astype(np.float32)
    if valid.size == 0:
        raise ValueError(f"No valid depth in release ROI {roi}")

    q1 = float(np.percentile(valid, 25))
    q3 = float(np.percentile(valid, 75))
    iqr = max(q3 - q1, 1.0)
    lo = max(1.0, q1 - 1.5 * iqr)
    hi = min(float(valid_max - 1), q3 + 1.5 * iqr + RELEASE_IQR_MARGIN_RAW)

    filtered_mask = valid_mask & (patch >= lo) & (patch <= hi)
    filtered = patch[filtered_mask].astype(np.float32)
    if filtered.size == 0:
        filtered_mask = valid_mask
        filtered = valid

    z_raw = float(np.percentile(filtered, RELEASE_SURFACE_PERCENTILE))
    candidate_mask = filtered_mask & (np.abs(patch - z_raw) <= RELEASE_DEPTH_EPSILON_RAW)
    ys, xs = np.nonzero(candidate_mask)
    if ys.size == 0:
        ys, xs = np.nonzero(filtered_mask)
        z_raw = float(np.percentile(filtered, RELEASE_SURFACE_PERCENTILE))

    u_img = float(x0 + np.mean(xs))
    v_img = float(y0 + np.mean(ys))
    return u_img, v_img, z_raw


def save_release_debug_visualization(depth_img, roi, u_img, v_img, z_raw, out_path):
    valid = depth_img[(depth_img > 0) & (depth_img < 65000)].astype(np.float32)
    if valid.size > 0:
        lo, hi = np.percentile(valid, [2, 98])
        depth_norm = np.clip(
            (depth_img.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6),
            0,
            1,
        )
    else:
        depth_norm = np.zeros_like(depth_img, dtype=np.float32)

    vis_gray = (depth_norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis_gray, cv2.COLORMAP_TURBO)
    invalid_mask = ~((depth_img > 0) & (depth_img < 65000))
    vis[invalid_mask] = 0

    x0, y0, x1, y1 = roi
    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)

    ui = int(round(u_img))
    vi = int(round(v_img))
    cv2.circle(vis, (ui, vi), 6, (0, 0, 255), -1)
    cv2.circle(vis, (ui, vi), 12, (255, 255, 255), 2)

    label = f"release z_raw={z_raw:.0f} ({z_raw * DEPTH_SCALE_TO_M:.3f}m)"
    cv2.putText(
        vis,
        label,
        (max(10, x0), max(28, y0 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"pixel=({ui}, {vi}) roi=({x0},{y0})-({x1},{y1})",
        (10, vis.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), vis)


class HoverService:
    def __init__(self):
        self.lock = threading.Lock()
        self.piper = C_PiperInterface_V2(CAN_NAME, False)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

    def gripper_call(self, cmd: str, sock_timeout: float = 5.0, **kwargs):
        if not os.path.exists(GRIPPER_SOCKET):
            raise RuntimeError(
                "gripper server is not running. start pika services first: python3 /home/swfu/ws/launch_pika_services.py"
            )

        req = {
            "id": str(uuid.uuid4()),
            "cmd": cmd,
            **kwargs,
        }

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(sock_timeout)
                client.connect(GRIPPER_SOCKET)
                send_one_line(client, req)
                raw = recv_one_line(client)
        except OSError:
            raise RuntimeError(
                "failed to connect gripper server. start pika services first: python3 /home/swfu/ws/launch_pika_services.py"
            )

        if not raw:
            raise RuntimeError("empty response from gripper server")

        resp = json.loads(raw)
        if not resp.get("ok", False):
            raise RuntimeError(f"gripper error: {resp.get('code')} {resp.get('error')}")
        return resp

    def photo_call(self, cmd: str, sock_timeout: float = 8.0, **kwargs):
        if not os.path.exists(PHOTO_SOCKET):
            raise RuntimeError(
                "photo server is not running. start photo services first: python3 /home/swfu/ws/main.py"
            )

        req = {
            "id": str(uuid.uuid4()),
            "cmd": cmd,
            **kwargs,
        }

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(sock_timeout)
                client.connect(PHOTO_SOCKET)
                send_one_line(client, req)
                raw = recv_one_line(client)
        except OSError:
            raise RuntimeError(
                "failed to connect photo server. start photo services first: python3 /home/swfu/ws/main.py"
            )

        if not raw:
            raise RuntimeError("empty response from photo server")

        resp = json.loads(raw)
        if not resp.get("ok", False):
            raise RuntimeError(f"photo error: {resp.get('code')} {resp.get('error')}")
        return resp

    def status(self):
        current_xyz_m, current_rpy_deg = current_pose_from_piper(self.piper)
        return {
            "can_name": CAN_NAME,
            "current_pose": {
                "xyz_m": current_xyz_m.tolist(),
                "rpy_deg": current_rpy_deg.tolist(),
            },
        }

    def execute_grasp(self, req: dict):
        level = req.get("Level", req.get("level"))
        pre_release_pose_cmd = resolve_pre_release_pose_cmd(
            level,
            req.get("pre_release_pose_cmd", req.get("release_pose_cmd")),
            req.get("release_joint_cmd"),
        )
        formal_release_pose_cmd = resolve_formal_release_pose_cmd(
            level,
            req.get("formal_release_pose_cmd"),
            req.get("formal_release_joint_cmd"),
        )


        # ROI 自适应深度
        with open(INFER_JSON, "r", encoding="utf-8") as file_obj:
            infer = json.load(file_obj)
        depth = np.load(DEPTH_NPY).astype(np.float32)
        with open(CAPTURE_POSE_JSON, "r", encoding="utf-8") as file_obj:
            capture_pose = json.load(file_obj)

        u_img, v_img = map_input_to_fullres(
            infer["u_300"],
            infer["v_300"],
            infer["crop_x0"],
            infer["crop_y0"],
            infer["crop_side"],
            input_side=infer.get("input_side", 300),
        )
        z_raw = robust_depth_at(depth, u_img, v_img, radius=3)
        p_cam = pixel_to_camera_xyz(u_img, v_img, z_raw)

        cap_flange_xyz_m, cap_flange_rpy_deg = pose_json_to_xyz_rpy_deg(capture_pose)
        cur_flange_xyz_m, cur_flange_rpy_deg = current_pose_from_piper(self.piper)

        base_rotation = rpy_deg_to_matrix(
            cap_flange_rpy_deg[0],
            cap_flange_rpy_deg[1],
            cap_flange_rpy_deg[2],
        )
        t_base_flange = make_transform(base_rotation, cap_flange_xyz_m)
        t_base_cam = t_base_flange @ T_FLANGE_TCP @ T_TCP_CAM
        target_tcp_xyz_m = transform_point(t_base_cam, p_cam)
        hover_tcp_xyz_m = target_tcp_xyz_m.copy()
        hover_tcp_xyz_m[2] += HOVER_Z_OFFSET_M

        keep_rpy_deg = cap_flange_rpy_deg.copy()
        cur_tcp_xyz_m = flange_to_tcp_xyz(cur_flange_xyz_m, cur_flange_rpy_deg)
        hover_flange_xyz_m = tcp_target_to_flange_cmd(hover_tcp_xyz_m, keep_rpy_deg)

        x_raw, y_raw, z_cmd, rx_raw, ry_raw, rz_raw = pose_to_command_units(
            hover_flange_xyz_m, keep_rpy_deg
        )
        self.gripper_call("enable")

        # 移动到夹取点
        self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        z_cmd = z_cmd if z_cmd>Z_PROTECT else Z_PROTECT     # 避免撞桌
        print("夹取点：", x_raw, y_raw, z_cmd, rx_raw, ry_raw, rz_raw)
        self.piper.EndPoseCtrl(x_raw, y_raw, z_cmd, rx_raw, ry_raw, rz_raw)
        close_resp = self.gripper_call("open")
        time.sleep(APPROACH_WAIT_SEC)
        print("夹取点", self.piper.GetArmStatus(), "\n\n")

        close_resp = self.gripper_call("close")
        time.sleep(CLOSE_WAIT_SEC)

        # 夹取点上抬
        print("夹取上抬：", x_raw, y_raw, z_cmd + 100000, rx_raw, ry_raw, rz_raw)
        self.piper.EndPoseCtrl(x_raw, y_raw, z_cmd + 100000, rx_raw, ry_raw, rz_raw)
        time.sleep(APPROACH_WAIT_SEC)
        print("夹取上抬", self.piper.GetArmStatus(), "\n\n")

        # 先移动到每个 Level 对应的固定预放置位，再拍照估计最终落点。
        print("预放置点：", *pre_release_pose_cmd)
        self.piper.MotionCtrl_2(0x01, 0x00, 40, 0x00)
        self.piper.EndPoseCtrl(*pre_release_pose_cmd)
        time.sleep(RELEASE_WAIT_SEC)
        print("预放置点", self.piper.GetArmStatus(), "\n\n")

        release_capture_resp = self.photo_call("capture")
        release_depth = np.load(DEPTH_NPY).astype(np.float32)
        with open(CAPTURE_POSE_JSON, "r", encoding="utf-8") as file_obj:
            release_capture_pose = json.load(file_obj)
        release_roi = resolve_release_roi(level, release_depth.shape)
        release_u_img, release_v_img, release_z_raw = filtered_surface_depth_in_roi(
            release_depth, release_roi
        )
        save_release_debug_visualization(
            release_depth,
            release_roi,
            release_u_img,
            release_v_img,
            release_z_raw,
            RELEASE_DEBUG_PNG,
        )

        release_p_cam = pixel_to_camera_xyz(release_u_img, release_v_img, release_z_raw)
        release_flange_xyz_m, release_flange_rpy_deg = pose_json_to_xyz_rpy_deg(
            release_capture_pose
        )
        release_rotation = rpy_deg_to_matrix(
            release_flange_rpy_deg[0],
            release_flange_rpy_deg[1],
            release_flange_rpy_deg[2],
        )
        t_base_release_flange = make_transform(release_rotation, release_flange_xyz_m)
        t_base_release_cam = t_base_release_flange @ T_FLANGE_TCP @ T_TCP_CAM
        release_surface_xyz_m = transform_point(t_base_release_cam, release_p_cam)

        formal_release_flange_xyz_m, formal_release_rpy_deg = pose_cmd_to_xyz_rpy_deg(
            formal_release_pose_cmd
        )
        release_target_tcp_xyz_m = flange_to_tcp_xyz(
            formal_release_flange_xyz_m, formal_release_rpy_deg
        )
        release_target_tcp_xyz_m[2] = release_surface_xyz_m[2] + RELEASE_Z_CLEARANCE_M

        release_target_flange_xyz_m = tcp_target_to_flange_cmd(
            release_target_tcp_xyz_m, formal_release_rpy_deg
        )
        (
            release_x_raw,
            release_y_raw,
            release_z_cmd,
            release_rx_raw,
            release_ry_raw,
            release_rz_raw,
        ) = pose_to_command_units(release_target_flange_xyz_m, formal_release_rpy_deg)
        release_z_cmd = release_z_cmd if release_z_cmd > Z_PROTECT else Z_PROTECT

        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        print("正式放置点", 
            release_x_raw,
            release_y_raw,
            release_z_cmd,
            release_rx_raw,
            release_ry_raw,
            release_rz_raw)
        
        self.piper.EndPoseCtrl(
            release_x_raw,
            release_y_raw,
            release_z_cmd,
            release_rx_raw,
            release_ry_raw,
            release_rz_raw,
        )
        time.sleep(1.5)
        print("正式放置点", self.piper.GetArmStatus(), "\n\n")

        open_resp = self.gripper_call("open")

        time.sleep(1)
        print("预上抬：", 
            release_x_raw,
            release_y_raw,
            release_z_cmd + 50000,
            release_rx_raw,
            release_ry_raw,
            release_rz_raw)
        
        self.piper.EndPoseCtrl(
            release_x_raw,
            release_y_raw,
            release_z_cmd + 50000,
            release_rx_raw,
            release_ry_raw,
            release_rz_raw,
        )

        time.sleep(1)

        print("预上抬：", self.piper.GetArmStatus(), "\n\n")

        self.gripper_call("disable")

        return {
            "Level": level,
            "pre_release_pose_cmd": pre_release_pose_cmd,
            "formal_release_pose_cmd": formal_release_pose_cmd,
            "gripper_close": close_resp,
            "gripper_open": open_resp,
            "u_img": float(u_img),
            "v_img": float(v_img),
            "depth_raw": float(z_raw),
            "p_cam_m": p_cam.tolist(),
            "current_tcp_xyz_m": cur_tcp_xyz_m.tolist(),
            "target_tcp_xyz_m": target_tcp_xyz_m.tolist(),
            "hover_flange_xyz_m": hover_flange_xyz_m.tolist(),
            "release_roi": [int(value) for value in release_roi],
            "release_capture": release_capture_resp,
            "release_debug_png": str(RELEASE_DEBUG_PNG),
            "release_depth_raw": float(release_z_raw),
            "release_p_cam_m": release_p_cam.tolist(),
            "release_surface_xyz_m": release_surface_xyz_m.tolist(),
            "release_target_tcp_xyz_m": release_target_tcp_xyz_m.tolist(),
        }

    def handle(self, req: dict):
        req_id = req.get("id")
        cmd = req.get("cmd")

        with self.lock:
            try:
                if cmd == "status":
                    data = self.status()
                elif cmd == "grasp":
                    data = self.execute_grasp(req)
                else:
                    return {
                        "id": req_id,
                        "ok": False,
                        "code": "BAD_REQUEST",
                        "error": f"unknown cmd: {cmd}",
                    }

                return {
                    "id": req_id,
                    "ok": True,
                    **data,
                }
            except Exception as exc:
                return {
                    "id": req_id,
                    "ok": False,
                    "code": "EXCEPTION",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }


def main():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(8)

    svc = HoverService()
    print(f"[INFO] hover server listening at {SOCKET_PATH}")

    while True:
        conn, _ = server.accept()
        with conn:
            raw = recv_one_line(conn)
            if not raw:
                continue

            try:
                req = json.loads(raw)
            except Exception:
                send_one_line(
                    conn,
                    {
                        "id": None,
                        "ok": False,
                        "code": "BAD_JSON",
                        "error": "invalid json",
                    },
                )
                continue

            resp = svc.handle(req)
            send_one_line(conn, resp)


if __name__ == "__main__":
    main()
