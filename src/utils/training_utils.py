import os
import numpy as np
import torch
import torch.nn as nn
import time, datetime, shutil, sys, os
import glob
from torch.utils.tensorboard import SummaryWriter




def set_rand_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic= True
        torch.backends.cudnn.benchmark = False

'''
Log utils
'''

class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
def printENV():
    check_list = ['CUDA_VISIBLE_DEVICES']
    for name in check_list:
        if name in os.environ:
            print(name, os.environ[name])
        else:
            print(name, "Not find")

    sys.stdout.flush()
    
    
def save_log(args):
    basedir = args.basedir
    expname = args.expname
    # logs
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    date_str = datetime.datetime.now().strftime("%m%d-%H%M%S")
    filedir = 'log_' + ('train' if not (args.test_mode) else 'test')
    filedir += date_str
    log_dir = os.path.join(basedir, expname, filedir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    sys.stdout = Logger(log_dir, False, fname="log.out")
    sys.stderr = Logger(log_dir, False, fname="log.err")
    
    print(" ".join(sys.argv), flush=True)
    printENV()

    # files backup
    shutil.copyfile(args.config, os.path.join(basedir, expname, filedir, 'config.txt'))
    f = os.path.join(log_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    backup_dir = os.path.join(log_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(backup_dir + '/src', exist_ok=True)
    
    filelist = glob.glob('*.py')
    filelist += glob.glob('src/*.py')
    filelist += glob.glob('src/*/*.py')
    for filename in filelist:
        # shutil.copyfile('./' + filename, os.path.join(backup_dir, filename))
        shutil.copyfile('./' + filename, os.path.join(log_dir, filename.replace("/","_")))

    return filedir, writer

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        ret = []
        for i in range(0, inputs.shape[0], chunk):
            tmp_ret = fn(inputs[i:i+chunk])
            # if fn.eval_mode:
            #     ret.append(tmp_ret.detach())
            # else:
            ret.append(tmp_ret)
        if len(ret) > 0:
            return torch.cat(ret,  0)
        else:
            return torch.tensor([])
        # return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def batchify_func(fn, chunk, eval_mode = False):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        ret = []
        for i in range(0, inputs.shape[0], chunk):
            tmp_ret = fn(inputs[i:i+chunk])
            if eval_mode:
                ret.append(tmp_ret.detach())
            else:
                ret.append(tmp_ret)
        if len(ret) > 0:
            return torch.cat(ret,  0)
        else:
            return torch.tensor([])
        # return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret
