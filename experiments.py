import torch
from torch.backends import cudnn

from kflod import kfold
import numpy as np
import random
from torch.optim.adam import Adam
from resnet_attn import *
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.densenet import densenet121
from preprocessing import get_dataset3d
import sys


def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# AllAtn网络 86.18%
def expAllAtn(data_path):
    reset_rand()

    def model_opt():
        model = AllAtn()
        optm = Adam(model.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='AllAtn',
          device='cuda:0',
          deterministic=True
          )

# 基本的Resnet网络 93.42%
def expBasicResnet(data_path):
    reset_rand()

    def model_opt():
        model = BasicResnet()
        optm = Adam(model.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='BasicResnet',
          device='cuda:0',
          deterministic=True
          )

# 论文定义的网络 95.62%
# 我们建议使用具有 3x3 内核大小的 Residual Blocks 进行局部特征提取，并使用 Non-Local Blocks 来提取全局特征。Non-Local Block
# 能够在不使用大量参数的情况下提取全局特征。Non-Local Block 背后的关键思想是在相同特征映射上的特征之

def expLocalGlobal(data_path):
    reset_rand()

    def model_opt():
        model = LocalGlobalNetwork()
        optm = Adam(model.parameters())
        return model, optm

    kfold(data_path,
          64,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='LocalGlobalNetwork',
          device='cuda:0',
          deterministic=True
          )

# AllAtnBig网络 85.89%
def expAllAtnBig(data_path):
    reset_rand()

    def model_opt():
        model = AllAtnBig()
        optm = Adam(model.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='AllAtnBig',
          device='cuda:0',
          deterministic=True
          )

# resnet50网络 86.82%
def expResnetTrans(data_path):
    reset_rand()

    def model_opt():
        model = resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.fc.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='ResnetTrans',
          device='cuda:0',
          deterministic=True,
          dataset_func=get_dataset3d
          )

# densenet121网络 92.50%
def expDensenetTrans(data_path):
    reset_rand()

    def model_opt():
        model = densenet121(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.classifier.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='DensenetTrans',
          device='cuda:0',
          deterministic=True,
          dataset_func=get_dataset3d
          )

# resnet18 网络 86.41%
def expResnet18Trans(data_path):
    reset_rand()

    def model_opt():
        model = resnet18(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.fc.parameters())
        return model, optm

    kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='Resnet18Trans',
          device='cuda:1',
          deterministic=True,
          dataset_func=get_dataset3d
          )


def print_error():
    print(f'python <model_name> <data_path>')
    print('here is a list of experiments names:')
    for name in experiments.keys():
        print(name)


if __name__ == '__main__':
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    experiments = {
        'Resnet18Trans': expResnet18Trans,
        'ResnetTrans': expResnetTrans,
        'AllAtnBig': expAllAtnBig,
        'LocalGlobal': expLocalGlobal,
        'BasicResnet': expBasicResnet,
        'AllAtn': expAllAtn,
        'DensenetTrans': expDensenetTrans
    }
    if len(sys.argv) < 3:
        print('Error, we expect two arguments')
        print_error()

    else:
        exp_name = sys.argv[1]
        data_path = sys.argv[2]
        if exp_name not in experiments:
            print('Unknow experiment name')
            print_error()
        else:

            experiments[exp_name](data_path)

# 在cmd 运行 ： python experiments.py LocalGlobal E:\课程学习\大二\大二下机器学习\小项目\肺分类\lidc_img