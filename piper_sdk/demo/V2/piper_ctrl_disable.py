#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 使能机械臂
import time
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("pipercan0", False)
    piper.ConnectPort()
    piper.DisableArm(1)

    print("2")
    time.sleep(2)
    piper.DisableArm(2)

    print("3")
    time.sleep(2)
    piper.DisableArm(3)

    piper.DisableArm(4)
    piper.DisableArm(5)
    piper.DisableArm(6)

    print("失能成功!!!!")
    
