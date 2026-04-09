#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
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

ARM_ENABLE_TIMEOUT_SEC = 5.0
ARM_DISABLE_TIMEOUT_SEC = 5.0

GO_ZERO_SPEED_RATE = 50

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
                time.sleep(0.3)
                return

            if status.arm_status.arm_status != 0x00:
                raise RuntimeError(f"机械臂运行出错：{status}")

            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"机械臂等待超时，timeout={timeout}s, 当前状态={status}"
                )
            

    def _pose_cmd_to_raw(self, pose_cmd: list[int]) -> dict:
        return {
            "x": pose_cmd[0],
            "y": pose_cmd[1],
            "z": pose_cmd[2],
            "rx": pose_cmd[3],
            "ry": pose_cmd[4],
            "rz": pose_cmd[5],
        }

    def _parse_speed_rate(self, req: dict) -> int:
        speed_rate = req.get("speed_rate", PHOTO_POSE["speed_rate"])
        try:
            speed_rate = int(speed_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed_rate 必须是整数") from exc

        if speed_rate <= 0:
            raise ValueError("speed_rate 必须大于 0")
        return speed_rate

    def _parse_pose_cmd(self, req: dict) -> list[int]:
        pose_cmd = req.get("pose_cmd")
        if not isinstance(pose_cmd, list):
            raise ValueError("pose_cmd 必须是长度为 6 的列表")
        if len(pose_cmd) != 6:
            raise ValueError("pose_cmd 必须包含 6 个值")

        parsed = []
        for value in pose_cmd:
            if isinstance(value, bool):
                raise ValueError("pose_cmd 不能包含布尔值")
            try:
                parsed.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"pose_cmd 中存在非整数值: {value}") from exc
        return parsed

    def _parse_joint_delta(self, req: dict) -> tuple[int, float]:
        joint_index = req.get("joint_index")
        delta_deg = req.get("delta_deg")

        try:
            joint_index = int(joint_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("joint_index 必须是 1 到 6 的整数") from exc

        if joint_index < 1 or joint_index > 6:
            raise ValueError("joint_index 必须在 1 到 6 之间")

        try:
            delta_deg = float(delta_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("delta_deg 必须是数字") from exc

        if not math.isfinite(delta_deg):
            raise ValueError("delta_deg 必须是有限数字")

        return joint_index, delta_deg

    def _read_joint_values(self, wrapper, attr_name: str):
        payload = getattr(wrapper, attr_name, None)
        if payload is None:
            return None

        values = []
        for index in range(1, 7):
            field_name = f"joint_{index}"
            if not hasattr(payload, field_name):
                return None
            try:
                values.append(int(getattr(payload, field_name)))
            except (TypeError, ValueError):
                return None
        return values

    def current_joints(self) -> tuple[str, list[int]]:
        feedback = self.piper.GetArmJointMsgs()
        feedback_values = self._read_joint_values(feedback, "joint_state")
        if feedback_values is not None and getattr(feedback, "time_stamp", 0) > 0:
            return "feedback", feedback_values

        control = self.piper.GetArmJointCtrl()
        control_values = self._read_joint_values(control, "joint_ctrl")
        if control_values is not None and getattr(control, "time_stamp", 0) > 0:
            return "control", control_values

        raise RuntimeError("无法读取当前关节角，反馈和控制流都未准备好")

    def joints_to_degrees(self, joint_values: list[int]) -> list[float]:
        return [round(value / 1000.0, 3) for value in joint_values]

    def arm_enable_status(self) -> list[bool]:
        return list(self.piper.GetArmEnableStatus())

    def enable_arm(self, timeout: float = ARM_ENABLE_TIMEOUT_SEC) -> list[bool]:
        deadline = time.monotonic() + timeout
        while True:
            self.piper.EnablePiper()
            status = self.arm_enable_status()
            if status and all(status):
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"机械臂使能超时，当前状态={status}")
            time.sleep(0.05)

    def disable_arm(self, timeout: float = ARM_DISABLE_TIMEOUT_SEC) -> list[bool]:
        self.piper.DisableArm(1)

        print("2")
        time.sleep(2)
        self.piper.DisableArm(2)

        print("3")
        time.sleep(2)
        self.piper.DisableArm(3)

        self.piper.DisableArm(4)
        self.piper.DisableArm(5)
        self.piper.DisableArm(6)

    def current_pose(self):
        msg = self.piper.GetArmEndPoseMsgs().end_pose
        return {
            "x_m": msg.X_axis / 1000000.0,
            "y_m": msg.Y_axis / 1000000.0,
            "z_m": msg.Z_axis / 1000000.0,
            "rx_rad": msg.RX_axis / 1000.0 * math.pi / 180.0,
            "ry_rad": msg.RY_axis / 1000.0 * math.pi / 180.0,
            "rz_rad": msg.RZ_axis / 1000.0 * math.pi / 180.0,
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

    def move_pose(self, req: dict):
        pose_cmd = self._parse_pose_cmd(req)
        speed_rate = self._parse_speed_rate(req)

        self.piper.MotionCtrl_2(0x01, 0x00, speed_rate, 0x00)
        self.piper.EndPoseCtrl(*pose_cmd)
        self.waitMove()

        pose = self.current_pose()
        return {
            "speed_rate": speed_rate,
            "target_raw": self._pose_cmd_to_raw(pose_cmd),
            "pose": pose,
        }

    def move_joint_delta(self, req: dict):
        joint_index, delta_deg = self._parse_joint_delta(req)
        speed_rate = self._parse_speed_rate(req)

        before_source, before_joints_raw = self.current_joints()
        delta_raw = int(round(delta_deg * 1000.0))
        target_joints_raw = list(before_joints_raw)
        target_joints_raw[joint_index - 1] += delta_raw

        self.piper.MotionCtrl_2(0x01, 0x01, speed_rate, 0x00)
        self.piper.JointCtrl(*target_joints_raw)
        self.waitMove()

        after_source, after_joints_raw = self.current_joints()
        return {
            "joint_index": joint_index,
            "delta_deg": delta_deg,
            "delta_raw": delta_raw,
            "speed_rate": speed_rate,
            "before_joint_source": before_source,
            "after_joint_source": after_source,
            "before_joints_raw": before_joints_raw,
            "target_joints_raw": target_joints_raw,
            "after_joints_raw": after_joints_raw,
            "before_joints_deg": self.joints_to_degrees(before_joints_raw),
            "target_joints_deg": self.joints_to_degrees(target_joints_raw),
            "after_joints_deg": self.joints_to_degrees(after_joints_raw),
        }

    def go_zero(self):
        self.enable_arm()
        self.piper.MotionCtrl_2(0x01, 0x01, GO_ZERO_SPEED_RATE, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.waitMove()
        return {
            "speed_rate": GO_ZERO_SPEED_RATE,
            "target_joints_raw": [0, 0, 0, 0, 0, 0],
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

    def reset_and_disable(self):
        zero_resp = self.go_zero()
        enable_status_after_disable = self.disable_arm()
        return {
            "enable_status_after_disable": enable_status_after_disable,
            "go_zero": zero_resp,
        }

    def enable_and_reset(self):
        enable_status_after_enable = self.enable_arm()
        reset_resp = self.reset()
        return {
            "enable_status_after_enable": enable_status_after_enable,
            "reset": reset_resp,
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
                elif cmd == "move_pose":
                    data = self.move_pose(req)
                elif cmd == "move_joint_delta":
                    data = self.move_joint_delta(req)
                elif cmd == "reset_and_disable":
                    data = self.reset_and_disable()
                elif cmd == "enable_and_reset":
                    data = self.enable_and_reset()
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
            except ValueError as exc:
                return {
                    "id": req_id,
                    "ok": False,
                    "code": "BAD_REQUEST",
                    "error": str(exc),
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
