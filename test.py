import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("pipercan0", False)
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    
    piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
    piper.EndPoseCtrl(-37297, 253391, 252734, 180000, 17703, -81626) 
    piper.EndPoseCtrl(-145592, 193180, 228266, 180000, 29921, -52996) #2y

    piper.EndPoseCtrl(-223089, 296007, 173910, 180000, 23312, -52996)