import pyrealsense2 as rs
import torch

from myFind_handpoints import findhand
from my_function import *
from predict_hand import detect


class Keyboard:
    def __init__(self):
        self.extrinsic = None
        self.interval = 2.5
        self.shape = [6, 8]
        # 棋盘格世界坐标系坐标
        self.object_points = np.zeros((1, self.shape[0] * self.shape[1], 3), np.float32)
        self.object_points[0, :, :2] = np.mgrid[0:self.shape[0], 0:self.shape[1]].T.reshape(-1, 2)
        self.object_points *= self.interval
        # 相机内参数和畸变矩阵 相机为intel realsense D455
        self.intrinsics = np.array([[386.618, 0, 321.084], [0, 386.171, 241.835], [0, 0, 1]])
        self.distortions = np.array([-0.056022, 0.0656431, -0.000842583, -0.000429316, -0.0214383])
        # 相机配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # start streaming
        self.profile = self.pipeline.start(self.config)
        # 创建对齐对象
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    # 标定相机外参数
    def camera_calibration(self):
        print("Press s get extrinsics")
        while True:
            frames = self.pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue
            frame = np.asanyarray(frame.get_data())
            cv2.imshow('calibration', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                break
        cv2.destroyAllWindows()
        self.extrinsic = calibrate_ext(self.intrinsics, self.distortions, self.object_points, frame)
        if self.extrinsic == [0, 0]:
            print("cant detect chessboard")
            return
        print("extrinsics:\n", self.extrinsic)
        return

    # 开启键盘
    def keyboard(self):
        # 记录五指处于按压状态还是抬起状态
        press = [0, 0, 0, 0, 0]
        # 记录五指所处位置对应按键
        contex = [0, 0, 0, 0, 0]
        # 记录五指处于棋盘格位置
        ab = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        # 手指抬起落下 以及平移的信号量
        up = [-3, -3, -3, -3, -3]
        roll = [0, 0, 0, 0, 0]
        print("Keyboard start")
        print("press s to quit")
        print("press n to recalibrate extrinsics")
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_frame = 0.1 * np.asanyarray(aligned_depth_frame.get_data())
            img = frames.get_color_frame()
            img = np.asanyarray(img.get_data())

            # ===================== 深度学习嵌入 ====================================
            voc_config = 'cfg/hand.data'  # 模型相关配置文件
            # model_path = './hand-tiny_512-2021-02-19.pt'  # 检测模型路径
            model_path = './model_exp/hand_416-2021-02-20.pt'  # 检测模型路径
            model_cfg = 'yolo'  # yolo / yolo-tiny 模型结构

            img_size = 416  # 图像尺寸
            conf_thres = 0.5  # 检测置信度
            nms_thres = 0.6  # nms 阈值

            with torch.no_grad():  # 设置无梯度运行模型推理
                img, roi, xxyy = detect(
                    im0=img,
                    model_path=model_path,
                    cfg=model_cfg,
                    data_cfg=voc_config,
                    img_size=img_size,
                    conf_thres=conf_thres,
                    nms_thres=nms_thres,
                )
                if xxyy is not None:
                    img[xxyy[0]:xxyy[1], xxyy[2]:xxyy[3]], lmList = findhand(roi)

            hand = []
            if xxyy is not None:
                hand.append([lmList[4][0] + xxyy[2], lmList[4][1] + xxyy[0]])
                hand.append([lmList[8][0] + xxyy[2], lmList[8][1] + xxyy[0]])
                hand.append([lmList[12][0] + xxyy[2], lmList[12][1] + xxyy[0]])
                hand.append([lmList[16][0] + xxyy[2], lmList[16][1] + xxyy[0]])
                hand.append([lmList[20][0] + xxyy[2], lmList[20][1] + xxyy[0]])
                for i in hand:
                    if depth_frame[i[1]][i[0]] < 10:
                        continue
                for i in range(5):
                    if press[i] == 0:
                        point = calibrate_worldpoint(hand[i], self.extrinsic, self.intrinsics,
                                                     depth_frame[hand[i][1]][hand[i][0]])
                        if i == 0:
                            print(point)
                        judge1(point[0], point[1], point[2], up, press, contex, i, ab)
                    elif press[i] == 1:
                        point = calibrate_worldpoint(hand[i], self.extrinsic, self.intrinsics,
                                                     depth_frame[hand[i][1]][hand[i][0]])
                        if i == 0:
                            print(point)
                        judge2(point[0], point[1], point[2], press, contex, i, ab, up, roll)
            cv2.imshow("keyboard", img)
            k = cv2.waitKey(1) & 0xFF

            # 退出
            if k == ord('s'):
                print("Keyboard close")
                break

            # 重新标定相机外参数
            if k == ord('n'):
                self.extrinsic = calibrate_ext(self.intrinsics, self.distortions, self.object_points, img)
                print("New extrinsics:")
                print(self.extrinsic)
                if self.extrinsic == [0, 0]:
                    print("cant detect chessboard")
                    break
        cv2.destroyAllWindows()
        return

    def Keyboard_main(self):
        self.camera_calibration()
        self.keyboard()
        return


if __name__ == "__main__":
    K = Keyboard()
    K.Keyboard_main()
