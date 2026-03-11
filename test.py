import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("pipercan0", False)
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    
    # 在XOY平面上画正方形
    # 切换至MOVEP模式，移动到初始位置
    piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
    piper.EndPoseCtrl(-37297, 253391, 252734, 180000, 17703, -81626)