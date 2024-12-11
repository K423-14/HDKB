import time

from utils.datasets import *
from utils.parse_config import parse_data_cfg
from utils.torch_utils import select_device
from utils.utils import *
from yolov3 import Yolov3, Yolov3Tiny

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def process_data(img, img_size=416):  # 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def refine_hand_bbox(bbox, img_shape):
    height, width, _ = img_shape

    x1, y1, x2, y2 = bbox

    expand_w = (x2 - x1)
    expand_h = (y2 - y1)

    x1 -= expand_w * 0.06
    y1 -= expand_h * 0.1
    x2 += expand_w * 0.06
    y2 += expand_h * 0.1

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, width - 1))
    y2 = int(min(y2, height - 1))

    return x1, y1, x2, y2


def detect(
        im0,
        model_path,
        cfg,
        data_cfg,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
):
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)

    # Initialize model
    weights = model_path
    if "-tiny" in cfg:
        a_scalse = 416. / img_size
        anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
        anchors_new = [(int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse)) for j in range(len(anchors))]

        model = Yolov3Tiny(num_classes, anchors=anchors_new)

    else:
        a_scalse = 416. / img_size
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
        anchors_new = [(int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse)) for j in range(len(anchors))]
        model = Yolov3(num_classes, anchors=anchors_new)

    device = select_device()  # 运行硬件选择
    use_cuda = torch.cuda.is_available()
    # Load weights
    if os.access(weights, os.F_OK):  # 判断模型文件是否存在
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    model.to(device).eval()  # 模型模式设置为 eval

    # -------------------------------------------------

    t = time.time()
    img = process_data(im0, img_size)
    if use_cuda:
        torch.cuda.synchronize()
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    pred, _ = model(img)  # 图片检测
    if use_cuda:
        torch.cuda.synchronize()
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]  # nms
    if use_cuda:
        torch.cuda.synchronize()
    if detections is None or len(detections) == 0:
        return im0, None, None
    # Rescale boxes from 416 to true image size
    detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape).round()
    result = []
    for res in detections:
        result.append(
            (classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
    if use_cuda:
        torch.cuda.synchronize()

    # Draw bounding boxes and labels of detections
    for *xyxy, conf, cls_conf, cls in detections:
        label = '%s %.2f' % (classes[int(cls)], conf)

        xyxy = refine_hand_bbox(xyxy, im0.shape)
        xyxy = int(xyxy[0]), int(xyxy[1]) + 6, int(xyxy[2]), int(xyxy[3])
        if int(cls) == 0:
            plot_one_box(xyxy, im0, label=label, color=(15, 255, 95), line_thickness=3)
        else:
            plot_one_box(xyxy, im0, label=label, color=(15, 155, 255), line_thickness=3)

    roi = im0[(int(xyxy[1])):(int(xyxy[3])), (int(xyxy[0])):(int(xyxy[2]))]

    s2 = time.time()

    str_fps = ("{:.2f} Fps".format(1. / (s2 - t + 0.00001)))
    cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255), 4)
    cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 1)

    return im0, roi, [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])]


if __name__ == '__main__':
    voc_config = 'cfg/hand.data'  # 模型相关配置文件
    # model_path = './hand-tiny_512-2021-02-19.pt'  # 检测模型路径
    model_path = './model_exp/hand_416-2021-02-20.pt'  # 检测模型路径
    model_cfg = 'yolo'  # yolo / yolo-tiny 模型结构
    video_path = "./5e91004558868.mp4"  # 测试视频

    img_size = 416  # 图像尺寸
    conf_thres = 0.5  # 检测置信度
    nms_thres = 0.6  # nms 阈值

    with torch.no_grad():  # 设置无梯度运行模型推理
        detect(
            model_path=model_path,
            cfg=model_cfg,
            data_cfg=voc_config,
            img_size=img_size,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
        )
