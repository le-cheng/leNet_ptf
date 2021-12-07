from __future__ import print_function

import importlib
import logging
import os
import shutil
from time import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

import provider  # 数据增强
from dataset import ModelNetDataLoader


@hydra.main(config_path='config', config_name='cls')
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)# Using OmegaConf.set_struct, it is possible to prevent the creation of fields that do not exist
    global logger
    logger = logging.getLogger(__name__) # name
    print(OmegaConf.to_yaml(cfg))

    '''Hyper Parameter超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device}, GPU is NO.{str(cfg.gpu)} device')

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    TRAIN_DATASET = ModelNetDataLoader(
        root=DATA_PATH, npoint=cfg.num_point, 
        split='train' , normal_channel=cfg.normal)
    TEST_DATASET = ModelNetDataLoader(
        root=DATA_PATH, npoint=cfg.num_point, 
        split='test'  , normal_channel=cfg.normal)
    Train_DataLoader = DataLoader(
        dataset=TRAIN_DATASET , batch_size=cfg.batch_size, 
        shuffle=True  , num_workers=cfg.num_workers)
    Test_DataLoader =  DataLoader(
        dataset=TEST_DATASET  , batch_size=cfg.batch_size, 
        shuffle=False , num_workers=cfg.num_workers)

    
    '''MODEL LOADING'''
    logger.info('Load MODEL ...')
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(cfg.model.name)), '.')
    model = getattr(importlib.import_module('models.{}.model'.format(cfg.model.name)), 'PointTransformerCls')(cfg).cuda()
    # 数据并行,效率低，运算也不平衡，但是内存不够大可以用这种方法
    # model = nn.DataParallel(model.cuda())

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


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
    lossfn = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = MultiStepLR(
        optimizer, 
        milestones = [120,180], 
        gamma=cfg.scheduler_gamma
        )
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    global writer
    writer = SummaryWriter()
    t1 = time()
    for epoch in range(start_epoch,cfg.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epoch))
        train(model, Train_DataLoader, optimizer, epoch, lossfn)
        scheduler.step()
        instance_acc, class_acc = test(model, Test_DataLoader)

        if (instance_acc > best_instance_acc):
            best_instance_acc = instance_acc
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

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
        logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

        writer.add_scalar('Test_Acc', instance_acc, epoch)
        writer.add_scalar('Best_Acc', best_instance_acc, epoch)
        writer.add_scalar('ClassAcc', class_acc, epoch)
        writer.add_scalar('Best_ClassAcc', best_class_acc, epoch)

        global_epoch += 1

    logger.info('End of training...')
    t2 = time()
    logger.info('trian and eval model time is %.4f h'%((t2-t1)/3600))
    writer.close()
    return 0

def train(model, Train_DataLoader, optimizer, epoch, lossfn):
    model.train()
    correct = 0
    for batch_id, data in tqdm(enumerate(Train_DataLoader, 0), total=len(Train_DataLoader), smoothing=0.9, desc='Train_Epoch'):
        points, target = data
        points = points.data.numpy() #  (16, 1024, 6)
        points = provider.random_point_dropout(points)
        points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        
        # Compute prediction and loss
        pred = model(points)
        loss = lossfn(pred, target.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).cpu().sum().item()
    train_instance_acc = correct / len(Train_DataLoader.dataset)
    logger.info('Train Instance Accuracy: %f' % train_instance_acc)
    writer.add_scalar('train_Acc', train_instance_acc, epoch)

def test(model, test_loader, num_class=40):
    """验证
    Args:
        model : 网络
        loader : 数据
        num_class (int, optional): 分类数量. Defaults to 40.

    Returns:
        instance_acc，class_acc
    """
    model.eval()# 一定要model.eval()在推理之前调用方法以将 dropout 和批量归一化层设置为评估模式。否则会产生不一致的推理结果。
    logger.info('Load dataset ...')
    # mean_correct = []
    class_acc = np.zeros((num_class,3))
    with torch.no_grad():
        correct=0
        for j, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9, desc='Eval_Epoch'):
            # target = target[:, 0]
            points, target = points.cuda(), target[:, 0].cuda()
            pred = model(points)

            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            for cat in np.unique(target.cpu()):
                cat_idex = (target==cat)
                classacc = pred[cat_idex].eq(target[cat_idex].long().data).cpu().sum()

                class_acc[cat,0]+= classacc.item()
                class_acc[cat,1]+= cat_idex.sum()
            correct += pred.eq(target.view_as(pred)).cpu().sum()
        test_instance_acc=correct / len(test_loader.dataset)
    class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    return test_instance_acc, class_acc


if __name__ == '__main__':
    main()


