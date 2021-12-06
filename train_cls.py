"""
Author: Benny
Date: Nov 2019
"""
# import argparse
from time import time
import numpy as np
import os
from omegaconf import OmegaConf
import torch
# import datetime
import logging
from torch import nn
# from pathlib import Path
from tqdm import tqdm
# import syst
import provider
import importlib
import shutil
import hydra
# import omegaconf
from dataset import ModelNetDataLoader
from tensorboardX import SummaryWriter



@hydra.main(config_path='config', config_name='cls')
def main(args):
    OmegaConf.set_struct(args, False)# Using OmegaConf.set_struct, it is possible to prevent the creation of fields that do not exist

    '''Hyper Parameter超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
     

    global logger
    logger = logging.getLogger(__name__) # name

    
    print('\nargs:')
    print(OmegaConf.to_yaml(args))

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('modelnet40_normal_resampled/')

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=16)

    
    '''MODEL LOADING'''
    logger.info('Load MODEL ...')
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    # 从model获取PointTransformerCls，传入args，并将模型放到GPU上
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    
    # 数据并行,效率低，运算也不平衡，但是内存不够大可以用这种方法
    # classifier = nn.DataParallel(classifier.cuda())

    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9,
            weight_decay=0.0001
            )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones = [80,120,160], 
        gamma=args.scheduler_gamma
        )
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    writer = SummaryWriter()
    t1 = time()
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            # print(points.data.size())
            points = points.data.numpy()#(16, 1024, 6)
            # print(points.shape)
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        writer.add_scalar('train_Acc', train_instance_acc, epoch)


        with torch.no_grad():
            # instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            instance_acc, class_acc = test(classifier, testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            writer.add_scalar('Test_Acc', instance_acc, epoch)
            writer.add_scalar('Best_Acc', best_instance_acc, epoch)
            writer.add_scalar('ClassAcc', class_acc, epoch)
            writer.add_scalar('Best_ClassAcc', best_class_acc, epoch)

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    t2 = time()
    logger.info('trian and eval model time is %.4f'%((t2-t1)/60))
    writer.close()
    return 0

def test(model, loader, num_class=40):
    """验证

    Args:
        model : 网络
        loader : 数据
        num_class (int, optional): 分类数量. Defaults to 40.

    Returns:
        instance_acc，class_acc
    """
    logger.info('Load dataset ...')
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


if __name__ == '__main__':
    main()