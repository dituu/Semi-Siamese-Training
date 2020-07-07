from model import AttentionNet, MobileFaceNet
from tensorboardX import SummaryWriter
from prototype import Prototype
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch import optim


import os
import argparse
import numpy as np
import torch
import random
import lmdb_utils
import logging as logger
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')



def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def moving_average(probe, gallery, alpha):
    for param_probe, param_gallery in zip(probe.parameters(), gallery.parameters()):
        param_gallery.data =  alpha* param_gallery.data + (1 - alpha) * param_probe.detach().data


def trainlist_to_dict(source_file):
    trainfile_dict = {}
    with open(source_file, 'r') as infile:
        for line in infile:
            l = line.rstrip().lstrip()
            if len(l) > 0:
                lmdb_key, label = l.split(' ')
                label = int(label)
                if label not in trainfile_dict:
                    trainfile_dict[label] = {'lmdb_key':[],'num_images':0}
                trainfile_dict[label]['lmdb_key'].append(lmdb_key)
                trainfile_dict[label]['num_images'] += 1
    return trainfile_dict


def train_sample(train_dict, class_num, queue_size, last_id_list=False):
    all_id_list = range(0, class_num)
    # Make sure there is no overlap ids bewteen queue and curr batch.
    if last_id_list:
        tail_id= last_id_list[-queue_size:]
        non_overlap_id = list(set(all_id_list) - set(tail_id))
        head_id = random.sample(non_overlap_id, queue_size)
        remain_id = random.sample(list(set(all_id_list) - set(head_id)),class_num - queue_size)
        head_id.extend(remain_id) 
        curr_id_list = head_id
    else:
        curr_id_list = random.sample(all_id_list, class_num)
    
    # For each ID, two images are randomly sampled
    curr_train_list =[]
    for index in curr_id_list:
        lmdb_key_list =  train_dict[index]['lmdb_key']
        if int(train_dict[index]['num_images'])>1:
            training_samples = random.sample(lmdb_key_list, 2)
            line = training_samples[0] +' ' + training_samples[1]
            curr_train_list.append(line+' '+str(index) +'\n')
        else:
            line = lmdb_key_list[0] + ' '+lmdb_key_list[0]+' ' +str(index)
            curr_train_list.append(line+ '\n')
    return curr_train_list,curr_id_list


def train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer, 
    criterion, cur_epoch, conf):
    db_size = len(data_loader)
    check_point_size = (db_size // 2)
    batch_idx = 0
    initial_lr = get_lr(optimizer)

    probe_net.train()
    gallery_net.eval().apply(train_bn)

    for batch_idx, (images, _ ) in enumerate(data_loader):
        batch_size = images.size(0)
        global_batch_idx = (cur_epoch - 1) * db_size + batch_idx

        # the label of current batch in prototype queue
        label = (torch.LongTensor([range(batch_size)]) + global_batch_idx * batch_size) % conf.queue_size
        label = label.squeeze().cuda()
        images = images.cuda()
        x1, x2 = torch.split(images, [3, 3], dim=1)  

        # random set inputs as probe or gallery
        x1_probe = probe_net(x1)
        with torch.no_grad():
            x2_gallery = gallery_net(x2)  
        x2_probe = probe_net(x2)
        with torch.no_grad():
            x1_gallery = gallery_net(x1)
        output1, output2  = prototype(x1_probe,x2_gallery,x2_probe,x1_gallery,label)

        # random set inputs as probe or gallery
        loss = criterion(output1, label) + criterion(output2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        moving_average(probe_net, gallery_net, conf.alpha)

        if batch_idx % conf.print_freq == 0:
            loss_val = loss.item()
            lr = get_lr(optimizer)
            logger.info('epoch %d, iter %d, lr %f, loss %f'  % (cur_epoch, batch_idx, lr, loss_val))
            conf.writer.add_scalar('Train_loss', loss_val, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)

    if cur_epoch % conf.save_freq == 0 :
        saved_name = ('{}_epoch_{}.pt'.format(conf.model_type,cur_epoch))
        torch.save(probe_net.state_dict(), os.path.join(conf.saved_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)

def train_sst(conf):
    if conf.model_type == 'attention':
        probe_net = AttentionNet(attention_stages=conf.attention_stages, dim=conf.feat_dim)
        gallery_net = AttentionNet(attention_stages=conf.attention_stages, dim=conf.feat_dim) 
    elif conf.model_type == 'mobilefacenet':
        probe_net = MobileFaceNet(conf.feat_dim)
        gallery_net = MobileFaceNet(conf.feat_dim) 
        
    moving_average(probe_net, gallery_net, 0)
    prototype = Prototype(conf.feat_dim, conf.queue_size, conf.scale,conf.margin, conf.loss_type).cuda()     
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(probe_net.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.lr_decay_epochs, gamma=0.1)
    probe_net = torch.nn.DataParallel(probe_net).cuda()
    gallery_net = torch.nn.DataParallel(gallery_net).cuda()

    train_dict = trainlist_to_dict(conf.source_file)

    for epoch in range(1, conf.epochs + 1):
        if epoch == 1:
            curr_train_list, curr_id_list = train_sample(train_dict, conf.class_num, conf.queue_size)
        else:
            curr_train_list, curr_id_list = train_sample(train_dict, conf.class_num, conf.queue_size, curr_id_list)
        data_loader = DataLoader(lmdb_utils.SingleLMDBDataset(conf.source_lmdb, curr_train_list, conf.key),
                                 conf.batch_size, shuffle = False, num_workers=4, drop_last = True)
        train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer, 
            criterion, epoch, conf)
        lr_schedule.step()



if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='train arcface on face database.')
    conf.add_argument('--key', type=int, default=None, help='you must give a key before training.')
    conf.add_argument("--train_db_dir", type=str, default='/export/home/data', help="input database name")
    conf.add_argument("--train_db_name", type=str, default='deepglint_unoverlap_part40', help="comma separated list of training database.")
    conf.add_argument("--train_file_dir", type=str, default='/export/home/data/deepglint_unoverlap_part40', help="input train file dir.")
    conf.add_argument("--train_file_name", type=str, default='deepglint_train_list.txt', help="input train file name.")
    conf.add_argument("--output_model_dir", type=str, default='./snapshot', help=" save model paths")
    conf.add_argument('--model_type',type=str, default='mobilefacenet',choices=['mobilefacenet','attention'], help='choose model_type')    
    conf.add_argument('--attention_stages', type=str, default='1,1,1', help="1,1,1; 2,6,2; 3,8,3 is more commen")
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    conf.add_argument('--queue_size', type=int, default=16384, help='number of prototype queue')
    conf.add_argument('--class_num', type=int, default=72778, help='number of categories')
    conf.add_argument('--loss_type', type=str, default='softmax',choices=['softmax','am_softmax','arc_softmax'], help="loss type, can be softmax, am or arc")
    conf.add_argument('--margin', type=float, default=0.0, help='loss margin ')
    conf.add_argument('--scale', type=float, default=30.0, help='scaling parameter ')
    conf.add_argument('--lr', type=float, default=0.05, help='initial learning rate.')
    conf.add_argument('--epochs', type=int, default=100, help='how many epochs you want to train.')
    conf.add_argument('--lr_decay_epochs', type=str, default='48,72,90', help='similar to step specified in caffe solver, but in epoch mechanism')
    conf.add_argument('--momentum', type=float, default=0.9, help='momentum')
    conf.add_argument('--alpha', type=float, default=0.999, help='weight of moving_average')
    conf.add_argument('--batch_size', type=int, default=128, help='batch size over all gpus.')
    conf.add_argument('--print_freq', type=int, default=100, help='frequency of displaying current training state.')
    conf.add_argument('--save_freq', type=int, default=1, help='frequency of saving current training state.')
    args = conf.parse_args()
    args.lr_decay_epochs = [int(p) for p in args.lr_decay_epochs.split(',')]
    args.attention_stages = [int(s) for s in args.attention_stages.split(',')]
    args.source_file = os.path.join(args.train_file_dir, args.train_file_name)
    args.source_lmdb = os.path.join(args.train_db_dir, args.train_db_name)

    subdir =datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
    loss_type=args.loss_type
    args.saved_dir = os.path.join(args.output_model_dir,loss_type,subdir)
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    writer = SummaryWriter(log_dir=args.saved_dir)
    args.writer = writer
    logger.info('Start optimization.')
    logger.info(args)
    train_sst(args)
    logger.info('Optimization done!')
