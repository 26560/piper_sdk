#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path

from piper_sdk import *


SOCKET_PATH = "/tmp/piper_photo_pose.sock"
CAN_NAME = "pipercan0"
CAPTURE_POSE_JSON = Path("/home/swfu/ws/piper_ggcnn/ggcnn/captures/pose_0000.json")

PHOTO_POSE = {
    "X": -37297,
    "Y": 253391,
    "Z": 252734,
    "RX": 180000,
    "RY": 25000,
    "RZ": -81626,
    "speed_rate": 20,
    "settle_sec": 3.0,
}


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


def save_json_atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, ensure_ascii=False, indent=2)
    tmp.replace(path)


class PhotoPoseService:
    def __init__(self):
        self.lock = threading.Lock()
        self.piper = C_PiperInterface_V2(CAN_NAME, False)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        if hasattr(self.piper, "EnableFilterAbnormalData"):
            try:
                self.piper.EnableFilterAbnormalData()
            except Exception:
                pass
        
    def waitMove(self, timeout=10.0, interval=0.3):
        deadline = time.monotonic() + timeout

        while True:
            time.sleep(interval)    #TODO: advise change to tiemstamp
            status = self.piper.GetArmStatus()

            if status.arm_status.motion_status == 0x00:
                print(f"{status.arm_status.motion_status}")
                return

            if status.arm_status.arm_status != 0x00:
                raise RuntimeError(f"机械臂运行出错：{status}")

            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"机械臂等待超时，timeout={timeout}s, 当前状态={status}"
                )
            

    def current_pose(self):
        msg = self.piper.GetArmEndPoseMsgs().end_pose
        return {
            "x_m": msg.X_axis / 1000000.0,
            "y_m": msg.Y_axis / 1000000.0,
            "z_m": msg.Z_axis / 1000000.0,
            "rx_rad": msg.RX_axis / 1000.0 * 3.141592653589793 / 180.0,
            "ry_rad": msg.RY_axis / 1000.0 * 3.141592653589793 / 180.0,
            "rz_rad": msg.RZ_axis / 1000.0 * 3.141592653589793 / 180.0,
            "raw": {
                "x": int(msg.X_axis),
                "y": int(msg.Y_axis),
                "z": int(msg.Z_axis),
                "rx": int(msg.RX_axis),
                "ry": int(msg.RY_axis),
                "rz": int(msg.RZ_axis),
            },
            "worker_capture_timestamp": time.time(),
            "euler_order": "xyz",
        }

    def capture_pose(self):
        return {
            "pose": self.current_pose(),
        }

    def reset(self):
        self.piper.MotionCtrl_2(0x01, 0x00, PHOTO_POSE["speed_rate"], 0x00)
        self.piper.EndPoseCtrl(
            PHOTO_POSE["X"],
            PHOTO_POSE["Y"],
            PHOTO_POSE["Z"],
            PHOTO_POSE["RX"],
            PHOTO_POSE["RY"],
            PHOTO_POSE["RZ"],
        )
        
        self.waitMove()

        pose = self.current_pose()
        pose["target_raw"] = {
            "x": PHOTO_POSE["X"],
            "y": PHOTO_POSE["Y"],
            "z": PHOTO_POSE["Z"],
            "rx": PHOTO_POSE["RX"],
            "ry": PHOTO_POSE["RY"],
            "rz": PHOTO_POSE["RZ"],
        }
        save_json_atomic(CAPTURE_POSE_JSON, pose)
        return {
            "pose_json": str(CAPTURE_POSE_JSON),
            "pose": pose,
        }

    def status(self):
        return {
            "can_name": CAN_NAME,
            "pose_json": str(CAPTURE_POSE_JSON),
            "current_pose": self.current_pose(),
        }

    def handle(self, req: dict):
        req_id = req.get("id")
        cmd = req.get("cmd")

        with self.lock:
            try:
                if cmd == "status":
                    data = self.status()
                elif cmd == "capture_pose":
                    data = self.capture_pose()
                elif cmd == "reset":
                    data = self.reset()
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

    svc = PhotoPoseService()
    print(f"[INFO] photo pose server listening at {SOCKET_PATH}")

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
