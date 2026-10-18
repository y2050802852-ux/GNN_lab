import argparse
import json,yaml
import logging,sys,os
import random,numpy as np
import torch
import copy
import pandas as pd



def load_config(yaml_path,cli_overrides=None):
    """
    以 YAML 为基础， 使用命令行参数覆盖配置
    
    支持两种覆盖格式：
    1. ['lr=5e-2', 'model=gat'],
    2. {'lr':5e-2, 'model':'gat'}
    返回合并后的配置字典
    """
    with open(yaml_path,'r', encoding='utf-8') as f:
        yaml_cfg = yaml.safe_load(f) or {}

    if not isinstance(yaml_cfg,dict):
        raise ValueError(
            f'配置文件顶层必须是字典: {yaml_path}'
        )
    cfg = copy.deepcopy(yaml_cfg)

    for key_path, value in (cli_overrides or {}).items():
        keys = key_path.split(".") # "optimizer.lr":0.05 -> keys = ['optimizer', 'lr']
        cur = cfg

        # 定位嵌套字段的父字典
        for key in keys[:-1]:
            if key not in cur:
                raise KeyError(
                    f"配置路径 {key_path} 不存在：缺少{key!r}"
                )
            if not isinstance(cur[key], dict):
                raise ValueError(
                    f"{key!r} 不是字典，无法继续访问{key_path!r}"
                )
            cur = cur[key]  # 到达最后一层 : cur = {'lr': 0.01, ...}

        if keys[-1] not in cur:
            raise KeyError(f"配置字段 {key_path!r} 不存在")

        cur[keys[-1]] = value
    
    return cfg


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a Graph Neural Network"
    )
    # 模型参数
    parser.add_argument(
        '--model',
        type=str,
        choices=['gcn', 'sage', 'gat', 'gin'],
        default=None,
        help='Model name'
    )
    parser.add_argument(
        '--hidden',
        type=int,
        default=None,
        help='Hidden dimension',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help="Dropout probability"
    )
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Leanring rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )

    # # 支持重复输入
    # parser.add_argument(
    #     '--set',
    #     dest='overrides',
    #     action='append',
    #     default=[],
    #     metavar='KEY=VALUE',
    #     help='Override any existing YAML field',
    # )

    return parser

def setup_logger(out_dir, name='train'):
    """双输出：
        1.控制台
        2.文件
    """
    os.makedirs(out_dir,exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # 防重复注册
    logger.propagate = False
    out_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(out_format)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(out_dir,'train_log'), encoding='utf-8')
    # FileHandler: 把日志写入文件
    fh.setFormatter(out_format)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def seed_init(cfg):
    random.seed(cfg['randomseed'])
    np.random.seed(cfg['npseed'])
    torch.manual_seed(cfg['torchseed'])


# def get_overrides(args):
#     return {
#         key: value
#         for key ,value in vars(args).items()
#         if value is not None
#     }

def get_overrides(args):
    overrides = {}

    # 收集直接通过命令行指定的参数
    for key in ("model", "hidden", "dropout", "epochs", "lr", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    # 收集 --overrides 传入的 key=value 参数
    for item in (getattr(args, "overrides", None) or []):
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(
                f"无效的覆盖参数 {item!r}，应使用 key=value 格式"
            )
        overrides[key] = value

    return overrides


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_result_csv(result,out_dir,name,logger):
    result = pd.DataFrame(result)
    os.makedirs(out_dir,exist_ok=True)
    csv_path = os.path.join(out_dir,f'metrics_{name}.csv')
    result.to_csv(csv_path,index=False)
    logger.info(f"result has been saved to path: {csv_path}")
