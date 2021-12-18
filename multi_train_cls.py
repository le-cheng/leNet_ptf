from __future__ import print_function

import importlib
import logging
import os
import shutil
from time import time

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#########################################################
import torch.multiprocessing as mp
from numpy import random
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import provider  # 数据增强
from dataset import ModelNetDataLoader


# def cleanup():
#     dist.destroy_process_group()

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
#########################################################

@hydra.main(config_path='config', config_name='cls')
def mmmm(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)# Using OmegaConf.set_struct, it is possible to prevent the creation of fields that do not exist
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    '''CUDA Configuration'''

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.gpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #########################################################
    cfg.DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model_copy.py'.format(cfg.model.name)), '.')

    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    port = find_free_port()
    cfg.dist_url = f"tcp://localhost:{port}"
    cfg.ngpus_per_node = torch.cuda.device_count()
    cfg.world_size = cfg.ngpus_per_node * cfg.nodes   
    # print(cfg.world_size)   
    mp.spawn(main, nprocs=cfg.ngpus_per_node, args=(cfg,))   
    # PyTorch提供了mp.spawn来在一个节点启动该节点所有进程，每个进程运行train(i, args)，
    # 其中i从0到args.gpus - 1。
    #########################################################

from logging import handlers


def get_logger(filename='train.log'):
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s [line %(lineno)d] %(process)d] %(message)s"
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    th = handlers.TimedRotatingFileHandler(filename=filename,when='D',backupCount=3,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
    th.setFormatter(logging.Formatter(fmt))#设置文件里写入的格式
    # logger.addHandler(sh) #把对象加到logger里
    logger.addHandler(th)
    return logger

def is_main_process():
    return get_rank() == 0  
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
import torch.distributed as dist
import torch.nn as nn


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value = value/world_size

        return value


def main(gpu, cfg):
    if gpu == 0:
        # print(__file__)
        # print(os.getcwd())
        global logger,writer
        writer = SummaryWriter()
        logger = get_logger()
        # logger = logging.getLogger(__name__)
        logger.info("=> start creating model ...")
        # logger.debug("Do something")
        # logger.warning("Something maybe fail.")
        # logger.info("Finish")
    
    # global logger
    # logger = logging.getLogger(__name__) # name
    # print(OmegaConf.to_yaml(cfg))
    ############################################################
    rank = cfg.nr * cfg.ngpus_per_node + gpu	             
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method=cfg.dist_url,  #这里我们设置的是env://，指的是环境变量初始化方式，需要在环境变量中配置4个参数：MASTER_PORT，MASTER_ADDR，WORLD_SIZE，RANK，前面两个参数我们已经配置，后面两个参数也可以通过dist.init_process_group函数中world_size和rank参数配置。                                 
    	world_size=cfg.world_size,                              
    	rank=rank)                                                           
    ############################################################

    ############################################################
    torch.cuda.set_device(gpu)
    # cfg.batch_size = int(cfg.batch_size / cfg.ngpus_per_node)
    ############################################################

    '''Hyper Parameter超参数'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    # # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f'Using {device}, GPU is NO.{str(cfg.gpu)} device')

    '''DATA LOADING'''
    manual_seed = 1234
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    if gpu == 0:
        logger.info('Load dataset ...')
    # DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    TRAIN_DATASET = ModelNetDataLoader(
        root=cfg.DATA_PATH, npoint=cfg.num_point, 
        split='train' , normal_channel=cfg.normal, 
        classes=cfg.num_class, uniform=cfg.uniform)
    TEST_DATASET = ModelNetDataLoader(
        root=cfg.DATA_PATH, npoint=cfg.num_point, 
        split='test'  , normal_channel=cfg.normal, 
        classes=cfg.num_class, uniform=cfg.uniform)
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
    test_sampler = torch.utils.data.distributed.DistributedSampler(TEST_DATASET)
    ################################################################
    Train_DataLoader = DataLoader(
        dataset=TRAIN_DATASET, batch_size=cfg.batch_size, pin_memory=True,
        shuffle=(train_sampler is None)  , num_workers=cfg.num_workers, sampler=train_sampler)
    Test_DataLoader =  DataLoader(
        dataset=TEST_DATASET , batch_size=cfg.batch_size, pin_memory=True,
        shuffle=(test_sampler is None) , num_workers=cfg.num_workers, sampler=test_sampler)

    '''MODEL LOADING'''
    if gpu == 0:
        logger.info('Load MODEL ...')
    model = getattr(importlib.import_module('models.{}.model_copy'.format(cfg.model.name)), 'PointTransformerCls')(cfg).cuda()
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        if gpu == 0:
            logger.info('No existing model, starting training from scratch...')
        start_epoch = 0
    ###############################################################
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    ###############################################################
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.learning_rate, 
            momentum=0.9,
            weight_decay=0.0001
            )
    lossfn = torch.nn.CrossEntropyLoss().cuda()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = MultiStepLR(
        optimizer, 
        milestones = [cfg.epoch*0.6, cfg.epoch*0.8], 
        gamma=cfg.scheduler_gamma)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    if gpu == 0:
        logger.info('Start training...')
        t1 = time()
    for epoch in range(start_epoch,cfg.epoch):
        ###################################
        train_sampler.set_epoch(epoch)
        #############################
        if gpu == 0:
            logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epoch))

        train(model, Train_DataLoader, optimizer, epoch, lossfn)
        
        scheduler.step()
        if 1:
            instance_acc, class_acc = test(model, Test_DataLoader, cfg.num_class)
            if gpu == 0:
                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc

                if (instance_acc > best_instance_acc):
                    best_instance_acc = instance_acc
                logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
                logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
                if (instance_acc > best_instance_acc):
                    best_epoch = epoch + 1
                    logger.info('Save model...')
                    savepath = 'best_model.pth'
                    logger.info('Saving at %s'% savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)

                writer.add_scalar('Test_Acc', instance_acc, epoch)
                writer.add_scalar('Best_Acc', best_instance_acc, epoch)
                writer.add_scalar('ClassAcc', class_acc, epoch)
                writer.add_scalar('Best_ClassAcc', best_class_acc, epoch)

        global_epoch += 1

    if gpu == 0:
        logger.info('End of training...')
        t2 = time()
        logger.info('trian and eval model time is %.4f h'%((t2-t1)/3600))
        writer.close()
    # cleanup()
    return 0


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.data.numpy() #  (16, 1024, 6)
            self.next_input = provider.random_point_dropout(self.next_input)
            self.next_input[:,:, 0:3] = provider.random_scale_point_cloud(self.next_input[:,:, 0:3])
            self.next_input[:,:, 0:3] = provider.shift_point_cloud(self.next_input[:,:, 0:3])
            self.next_input = torch.Tensor(self.next_input)
            self.next_target = self.next_target[:, 0]
            # points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)

            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        points = self.next_input
        target = self.next_target
        self.preload()
        return points, target


def train(model, Train_DataLoader, optimizer, epoch, lossfn):
    model.train()
    correct = 0
    epoch_loss = 0 
    num_len = len(Train_DataLoader.dataset)
    if dist.get_rank() == 0:
        Train_DataLoader = tqdm(Train_DataLoader)

    # for batch_id, data in enumerate(Train_DataLoader, 0):
    prefetcher = data_prefetcher(Train_DataLoader)
    points, target = prefetcher.next()
    i = 0
    while points is not None:
        i += 1
        # points, target = data
        # points = points.data.numpy() #  (16, 1024, 6)
        # points = provider.random_point_dropout(points)
        # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
        # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        # points = torch.Tensor(points)
        # target = target[:, 0]
        # points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        # Compute prediction and loss
        pred = model(points)
        loss = lossfn(pred, target.long())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        #############################
        loss = reduce_value(loss, average=False)  
        # TODO: qian or hou
        ############################
        optimizer.step()
        # 计算
        pred = pred.argmax(dim=1, keepdim=True)
        ############################
        correct += pred.eq(target.view_as(pred)).sum()
        ############################
        epoch_loss+=loss

        points, target = prefetcher.next()
    
    train_instance_acc = correct / num_len
    train_instance_acc = reduce_value(train_instance_acc, average=False)
    epoch_loss = epoch_loss / len(Train_DataLoader)
    if dist.get_rank() == 0:
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        logger.info('Train Instance Loss: %f' % epoch_loss)
        writer.add_scalar('train_Acc', train_instance_acc, epoch)
        writer.add_scalar('train_Loss', epoch_loss, epoch)

def test(model, test_loader, num_class=40):
    model.eval()# 一定要model.eval()在推理之前调用方法以将 dropout 和批量归一化层设置为评估模式。
                # 否则会产生不一致的推理结果。
    # mean_correct = []
    class_acc = torch.zeros((num_class,3)).cuda()
    num_len = len(test_loader.dataset)
    with torch.no_grad():
        correct=0
        if dist.get_rank() == 0:
            test_loader = tqdm(test_loader)
        for j, (points, target) in enumerate(test_loader):
            points, target = points.cuda(non_blocking=True), target[:, 0].cuda(non_blocking=True)
            pred = model(points)
            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

            for cat in torch.unique(target):
                cat_idex = (target==cat)
                classacc = pred[cat_idex].eq(target[cat_idex].view_as(pred[cat_idex])).sum()
                class_acc[cat,0] += classacc
                class_acc[cat,1] += cat_idex.sum()
            # correct += pred.eq(target.view_as(pred)).cpu().sum()

        test_instance_acc=correct / num_len
        test_instance_acc = reduce_value(test_instance_acc, average=False)
        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc_t = torch.mean(class_acc[:,2])
        # class_acc_t = reduce_value(class_acc_t)
    return test_instance_acc, class_acc_t


if __name__ == '__main__':
    # 加载gc模块
    import gc

    # 垃圾回收gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()
    mmmm()


