import numpy as np
import cv2
import pyautogui


# 定义按钮类
class Button():
    def __init__(self, pos, text, size=[2.5, 2.5]):
        self.pos = pos
        self.size = size
        self.text = text


List = [["A", "B", "C", "D", "E"], ["F", "G", "H", "A", "J"], ["K", "L", "M", "N", "O"],
        ["P", "Q", "R", "S", "T"],
        ["U", "V", "W", "X", "Y"], ["Z", "esc", "esc", "esc", "esc"], ["esc", "backspace", "esc", "space", "esc"]]

# 存放所有按键
buttonList = []

for i, l in enumerate(List):
    for j, te in enumerate(l):
        buttonList.append(Button([j, i], te))


# 相机外参标定
def calibrate_ext(intrinsics, distortions, object_points, image):
    shape = (6, 8)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    gray = image
    succes, corners = cv2.findChessboardCorners(image=gray,
                                                patternSize=shape,
                                                flags=flags)
    if succes == True:
        image_points = corners
    else:
        return [0, 0]
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsics, distortions)
    if retval != True:
        print("error:input error")
        return
    rot = cv2.Rodrigues(rvec)[0]
    extrinsic = np.hstack([rot, tvec])
    extrinsic = np.vstack([extrinsic, [[0, 0, 0, 1]]])
    return extrinsic


# 输入图像坐标计算世界坐标系坐标
def calibrate_worldpoint(point_a, extrinsic_a, intrinsicsa, depth):
    point_a = np.append(point_a, 1)
    dis_intr_a = np.linalg.inv(intrinsicsa)

    dis_extr_a = np.linalg.inv(extrinsic_a)

    camera_point_a = dis_intr_a @ point_a
    camera_point_a = depth * camera_point_a

    camera_point_a = np.append(camera_point_a, 1)
    worldpoint = dis_extr_a @ camera_point_a
    return worldpoint


# 按压判断
def judge1(x, y, z, up, press, contex, i, ab):
    if z > -1 and z < 0:
        up[i] += 1
    elif z < -1:
        if up[i] > -3:
            up[i] -= 1
    if up[i] > 3:
        x = int(x)
        y = int(y)
        a = x // 2.5
        b = y // 2.5
        for button in buttonList:
            if a == button.pos[0] and b == button.pos[1]:
                pyautogui.keyDown(button.text)
                contex[i] = button.text
                press[i] = 1
                ab[i][0] = a
                ab[i][1] = b
                break
    return


# 抬起和平移判断
def judge2(x, y, z, press, contex, i, ab, up, roll):
    if z < -1:
        up[i] -= 1
    else:
        if up[i] < 3:
            up[i] += 1
    if up[i] < -3:
        pyautogui.keyUp(contex[i])
        roll[i] = 0
        contex[i] = 0
        press[i] = 0
        ab[i] = [0, 0]
    else:
        a = x // 2.5
        b = y // 2.5
        if a == ab[i][0] and b == ab[i][1]:
            if roll[i] < 0:
                roll[i] += 1
        else:
            roll[i] -= 1
        if roll[i] < -10:
            roll[i] = 0
            pyautogui.keyUp(contex[i])
            press[i] = 0
            contex[i] = 0
            ab[i] = [0, 0]
            for button in buttonList:
                if a == button.pos[0] and b == button.pos[1]:
                    pyautogui.keyDown(button.text)
                    press[i] = 1
                    contex[i] = button.text
                    ab[i][0] = a
                    ab[i][1] = b
                    break
    return
