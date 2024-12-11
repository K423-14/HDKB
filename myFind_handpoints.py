import argparse

import torch
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0

from hand_data_iter.datasets import draw_bd_handpose
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.rexnetv1 import ReXNetV1
from models.shufflenet import ShuffleNet
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import squeezenet1_1, squeezenet1_0
from utils.common_utils import *


def findhand(img):
    # 定义x和y列表
    lmlist = []

    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str,
                        default='./model_exp/shufflenet_v2_x2_0-size-256-model_epoch-97.pth',
                        help='model_path')  # 模型路径
    parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0',
                        help='''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,
            ''')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # 手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')  # GPU选择
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')  # 是否可视化图片

    # --------------------------------------------------------------------------
    ops = parser.parse_args()  # 解析添加参数
    # --------------------------------------------------------------------------

    vars(ops)

    # ---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    # ---------------------------------------------------------------- 构建模型

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)
    elif ops.model == "ReXNetV1":
        model_ = ReXNetV1(width_mult=1.0, depth_mult=1.0, num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()  # 设置为前向推断模式

    # 加载测试模型
    if os.access(ops.model_path, os.F_OK):  # checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)

    # ---------------------------------------------------------------- 预测图片

    with torch.no_grad():
        img_width = img.shape[1]
        img_height = img.shape[0]
        # 输入图片预处理
        try:
            img_ = cv2.resize(img, (ops.img_size[1], ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)
            pre_ = model_(img_.float())  # 模型推理
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            pts_hand = {}  # 构建关键点连线可视化结构
            for i in range(int(output.shape[0] / 2)):
                x = (output[i * 2 + 0] * float(img_width))
                y = (output[i * 2 + 1] * float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x": x,
                    "y": y,
                }
            draw_bd_handpose(img, pts_hand, 0, 0)  # 绘制关键点连线

            # ------------- 绘制关键点
            for i in range(int(output.shape[0] / 2)):
                x = (output[i * 2 + 0] * float(img_width))
                y = (output[i * 2 + 1] * float(img_height))

                cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

                lmlist.append([int(x), int(y)])

        except:
            print("pass")
        return img, lmlist
