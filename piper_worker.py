#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
import time
from pathlib import Path

from piper_sdk import *

CAN_NAME = "pipercan0"
BRIDGE_DIR = "/home/swfu/handeye_bridge"

# 官方 SDK 文档：
# X/Y/Z 单位 = 0.001 mm -> 1e-6 m
# RX/RY/RZ 单位 = 0.001 degree -> 1e-3 degree
POS_RAW_TO_M = 1e-6
ANG_RAW_TO_DEG = 1e-3

# 注意：SDK 只明确说明是 Euler angle representation，未公开明确写出组合顺序/参考系。
# 这里继续沿用 xyz，并在主程序里输出验证提示。
EULER_ORDER = "xyz"

# 仅接受“完全匹配”的固定格式，避免误抓其他数字字段。
POSE_TEXT_RE = re.compile(
    r"^\s*time\s*stamp\s*:\s*(?P<sdk_ts>[-+]?\d+(?:\.\d+)?)\s*[\r\n]+"
    r"\s*Hz\s*:\s*(?P<hz>[-+]?\d+(?:\.\d+)?)\s*[\r\n]+"
    r"\s*ArmMsgFeedBackEndPose\s*:\s*[\r\n]+"
    r"\s*X_axis\s*:\s*(?P<x>-?\d+)\s*[\r\n]+"
    r"\s*Y_axis\s*:\s*(?P<y>-?\d+)\s*[\r\n]+"
    r"\s*Z_axis\s*:\s*(?P<z>-?\d+)\s*[\r\n]+"
    r"\s*RX_axis\s*:\s*(?P<rx>-?\d+)\s*[\r\n]+"
    r"\s*RY_axis\s*:\s*(?P<ry>-?\d+)\s*[\r\n]+"
    r"\s*RZ_axis\s*:\s*(?P<rz>-?\d+)\s*$",
    re.MULTILINE,
)

REQ_POLL_SEC = 0.05
POSE_FRESH_TIMEOUT_SEC = 2.0
POSE_FRESH_GRACE_SEC = 0.05
# =======================

bridge = Path(BRIDGE_DIR)
req_dir = bridge / "requests"
pose_dir = bridge / "poses"
req_dir.mkdir(parents=True, exist_ok=True)
pose_dir.mkdir(parents=True, exist_ok=True)


def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_pose_text(text: str):
    match = POSE_TEXT_RE.match(text)
    if not match:
        debug_path = pose_dir / "last_pose_debug.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        raise RuntimeError(
            "GetArmEndPoseMsgs() 返回格式与预期不一致，已保存调试文本到 "
            f"{debug_path}"
        )

    groups = match.groupdict()
    sdk_ts = float(groups["sdk_ts"])
    hz = float(groups["hz"])
    x_raw = int(groups["x"])
    y_raw = int(groups["y"])
    z_raw = int(groups["z"])
    rx_raw = int(groups["rx"])
    ry_raw = int(groups["ry"])
    rz_raw = int(groups["rz"])

    pose = {
        "sdk_timestamp": sdk_ts,
        "hz": hz,
        "x_m": x_raw * POS_RAW_TO_M,
        "y_m": y_raw * POS_RAW_TO_M,
        "z_m": z_raw * POS_RAW_TO_M,
        "rx_rad": math.radians(rx_raw * ANG_RAW_TO_DEG),
        "ry_rad": math.radians(ry_raw * ANG_RAW_TO_DEG),
        "rz_rad": math.radians(rz_raw * ANG_RAW_TO_DEG),
        "raw": {
            "x": x_raw,
            "y": y_raw,
            "z": z_raw,
            "rx": rx_raw,
            "ry": ry_raw,
            "rz": rz_raw,
        },
        "unit_scale": {
            "pos_raw_to_m": POS_RAW_TO_M,
            "ang_raw_to_deg": ANG_RAW_TO_DEG,
        },
        "euler_order": EULER_ORDER,
        "text": text,
    }
    return pose


def read_fresh_pose_after_request(request_ts: float):
    last_pose = None
    deadline = time.time() + POSE_FRESH_TIMEOUT_SEC

    while time.time() < deadline:
        msg = piper.GetArmEndPoseMsgs()
        text = str(msg)
        try:
            pose = parse_pose_text(text)
        except Exception:
            raise

        pose["worker_capture_timestamp"] = time.time()
        last_pose = pose

        sdk_ts = pose["sdk_timestamp"]
        if sdk_ts >= request_ts - POSE_FRESH_GRACE_SEC:
            return pose

        time.sleep(0.01)

    if last_pose is None:
        raise RuntimeError("未能读取到任何末端位姿")

    raise TimeoutError(
        "读取到的位姿时间戳一直早于本次请求，"
        f"last_sdk_ts={last_pose['sdk_timestamp']:.6f}, request_ts={request_ts:.6f}"
    )


piper = C_PiperInterface_V2(CAN_NAME, False)
piper.ConnectPort()

# 官方 SDK 提供异常数据过滤接口，打开后能过滤一部分极端跳变。
if hasattr(piper, "EnableFilterAbnormalData"):
    try:
        piper.EnableFilterAbnormalData()
        print("[PIPER] EnableFilterAbnormalData() enabled")
    except Exception as exc:
        print(f"[PIPER] EnableFilterAbnormalData failed: {exc}")

print(f"[PIPER] connected: {CAN_NAME}")
print("[PIPER] waiting requests...")

while True:
    req_files = sorted(req_dir.glob("*.json"))
    for req_file in req_files:
        try:
            req = load_json(req_file)
        except Exception as exc:
            print(f"[PIPER] skip bad request {req_file}: {exc}")
            continue

        req_id = str(req.get("id", req_file.stem))
        request_ts = float(req.get("request_timestamp", req.get("ts", 0.0)))
        out_pose = pose_dir / f"{req_id}.json"

        if out_pose.exists():
            try:
                old = load_json(out_pose)
                if float(old.get("request_timestamp", -1.0)) == request_ts:
                    continue
            except Exception:
                pass

        try:
            pose = read_fresh_pose_after_request(request_ts)
        except Exception as exc:
            print(f"[PIPER] failed to capture pose {req_id}: {exc}")
            continue

        pose["id"] = req_id
        pose["request_timestamp"] = request_ts
        pose["request_file"] = str(req_file)

        save_json_atomic(out_pose, pose)
        print(f"[PIPER] saved pose {req_id}: {out_pose}")
        print(
            "[PIPER] raw pose:",
            pose["raw"],
            "sdk_ts=",
            pose["sdk_timestamp"],
            "capture_ts=",
            pose["worker_capture_timestamp"],
        )

    time.sleep(REQ_POLL_SEC)
