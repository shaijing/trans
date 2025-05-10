from datetime import datetime
import os
import random
import time
from typing import Union, Dict

import numpy as np
import torch
import warnings
from copy import deepcopy

from torch import nn
from torch.optim.lr_scheduler import LRScheduler


def setup_seed(seed):
    # random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def save_checkpoint(epoch: int,
                    module: Union[nn.Module, Dict],
                    optimizer: Union[torch.optim.Optimizer, Dict]=None,
                    schedule: LRScheduler = None,
                    path: str = 'checkpoints/',
                    safe: bool = True):
    """
    Parameters
    ----------
    epoch:
        当前epoch
    module:
        模型
    optimizer:
        优化器
    schedule:
        学习率调整器
    path:
        保存路径
    safe:
        是否安全模式
    """
    # Data dictionary to be saved
    if isinstance(module, nn.Module):
        module = module.state_dict()
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer.state_dict()
    if isinstance(schedule, LRScheduler):
        schedule = schedule.state_dict()
    data = {
        'epoch': epoch,
        # Current time (UNIX timestamp)
        'time': time.time(),
        # State dict the modules
        'model': module,
        # State dict the optimizers
        'optim': optimizer,
        # State dict the schedule
        'schedule': schedule
    }
    model_pth = f'{path}model_epoch_{epoch}.pth'
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(model_pth) and safe:
        # There's an old checkpoint. Rename it!
        old_path = model_pth + '.old'
        # old_path = path.replace('/', '.old/')

        # Remove the old checkpoint
        if os.path.exists(old_path):
            os.unlink(old_path)
        os.rename(model_pth, old_path)
    with open(model_pth, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

def load_checkpoint(path: str = "checkpoints/", epoch: int = None, verbose: bool = True):
    # If there's a checkpoint
    data = None
    if os.path.exists(path):
        if epoch is not None:
            model_pth = f'{path}model_epoch_{epoch}.pth'
            data = torch.load(model_pth)
            data['epoch'] = epoch
        else:
            checkpoint_files = os.listdir(path)
            # 过滤以'model_epoch_'开头且以'.pt'结尾的文件
            epoch_files = [file for file in checkpoint_files if
                           file.startswith('model_epoch_') and file.endswith('.pth')]
            if epoch_files:
                # 获取文件名中epoch部分，并将其转换为整数，然后找到最大值
                last_epoch = max([int(file.split('_')[2].split('.')[0]) for file in epoch_files])
                model_pth = f'{path}model_epoch_{last_epoch}.pth'
                # Load data
                data = torch.load(model_pth)
                print(f"最后一个epoch的文件为: model_epoch_{last_epoch}.pt")
            else:
                print("没有找到符合条件的epoch文件。")
        # Inform the user that we are loading the checkpoint
        if verbose:
            print(f"Loaded checkpoint saved at {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"Resuming from epoch {data['epoch']}")
    else:
        print("Checkpoints are not exists!")
    return data