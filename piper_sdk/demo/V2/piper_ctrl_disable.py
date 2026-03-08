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
    # while(piper.DisablePiper()):
    #     time.sleep(0.01)
    for i in range(1,7):
        print(i)
        time.sleep(3)
        piper.DisableArm(i)
    print("失能成功!!!!")
    