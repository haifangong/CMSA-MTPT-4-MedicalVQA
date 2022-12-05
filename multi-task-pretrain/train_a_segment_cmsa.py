import socket
import argparse
from datetime import datetime
import time
import os
import glob

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from dataloaders import abdomen_dataset, brain_tumor_dataset, chest_xray_dataset
from dataloaders import custom_transforms as trforms

from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101 
from model.classifier import Classifier
from model.segment_decoder import Decoder
from model.base_model import build_CMSA

import utils

from dataloaders import dictionary
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet34')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-dictionary_path', type=str, default='/data1/chenguanqi/Medical-VQA/MICCAI19-MedVQA/data_RAD/dictionary.pkl')
    parser.add_argument('-RAD_dir', type=str, default='/data1/chenguanqi/Medical-VQA/MICCAI19-MedVQA/data_RAD')
    parser.add_argument('-v_dim', type=int, default=512)
    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')
    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    # Choices of RNN models
    parser.add_argument('-rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-nepochs', type=int, default=100)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-train_fold', type=str, default='CMSA-Abdomen_ClipGrad-resnet34-glove')
    parser.add_argument('-run_id', type=int, default=-1)
    parser.add_argument('-T', type=int, default=2)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=20)

    parser.add_argument('-save_every', type=int, default=5)
    parser.add_argument('-log_every', type=int, default=25)
    parser.add_argument('-emb_init', type=str, default='glove', choices=['glove', 'biowordvec', 'biosentvec'])
    parser.add_argument('-self_att', action='store_true', default=False, help='Use Self Attention?')
    parser.add_argument('--use_spatial', action='store_true', default=False, help='Use spatial feature?')


    return parser.parse_args()

def softmax(x):
    # return np.exp(x) / sum(np.exp(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def adjust_learning_rate(optimizer, lr_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def adjust_learning_rate_DW(optimizer, scale):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dictionary_ = dictionary.Dictionary.load_from_file(args.dictionary_path)
    
    if args.backbone == 'resnet18':
        backbone = ResNet18(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet34':
        backbone = ResNet34(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet50':
        backbone = ResNet50(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet101':
        backbone = ResNet101(nInputChannels=3, os=32, pretrained=True)
    else:
        raise NotImplementedError

    # number of category for task
    a_classes = 14 # segmentation
    b_classes = 3 # if task is classification else 4 (for segmentation)
    c_classes = 2 # classification

    if args.backbone == 'resnet18' or args.backbone == 'resnet34':
        filters = [64, 64, 128, 256, 512]
    elif args.backbone == 'resnet50' or args.backbone == 'resnet101':
        filters = [64, 64*4, 128*4, 256*4, 512*4]
    else:
        raise NotImplementedError

    args.v_dim = filters[-1]

    a_decoder = Decoder(in_channels=filters[-1], filters=filters, n_classes=a_classes)
    cmsa = build_CMSA(dictionary=dictionary_, args=args)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold,'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', args.train_fold, 'run_'+str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    logger = open(os.path.join(save_dir, 'log.txt'), 'w')
    logger.write('optim: SGD \nlr=%.4f\nweight_decay=%.4f\nmomentum=%.4f\nupdate_lr_every=%d\nseed=%d\n' % 
        (args.lr, args.weight_decay, args.momentum, args.update_lr_every, args.seed))

    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    if args.resume_epoch == 0:
        print('Training from scratch...')
    else:
        backbone_resume_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(args.resume_epoch - 1) + '.pth')
        a_decoder_resume_path = os.path.join(save_dir, 'models', 'a_decoder_epoch-' + str(args.resume_epoch - 1) + '.pth')
        cmsa_resume_path = os.path.join(save_dir, 'models', 'cmsa_epoch-' + str(args.resume_epoch - 1) + '.pth')
        print('Initializing weights from: {}, epoch: {}...'.format(save_dir, args.resume_epoch))
        backbone.load_state_dict(torch.load(backbone_resume_path, map_location=lambda storage, loc: storage))
        a_decoder.load_state_dict(torch.load(a_decoder_resume_path, map_location=lambda storage, loc: storage))
        cmsa.load_state_dict(torch.load(cmsa_resume_path, map_location=lambda storage, loc: storage))
        
    torch.cuda.set_device(device=0)
    backbone.cuda()
    a_decoder.cuda()
    cmsa.cuda()


    cmsa_optim = optim.Adamax(filter(lambda p: p.requires_grad, cmsa.parameters()), lr=args.lr)
    backbone_optim = optim.Adamax(filter(lambda p: p.requires_grad, backbone.parameters()), lr=args.lr)
    a_decoder_optim = optim.Adamax(filter(lambda p: p.requires_grad, a_decoder.parameters()), lr=args.lr)
    
    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size+8, args.input_size+8)),
        trforms.RandomCrop(size=(args.input_size, args.input_size)),
        # trforms.RandomHorizontalFlip(),
        trforms.RandomRotate(degree=15),
        # trforms.RandomRotateOrthogonal(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    a_trainset = abdomen_dataset.AbdomenDataset(dictionary=dictionary_, question_len=12, mode='train', transform=composed_transforms_tr, return_size=False, seed=args.seed)
    a_valset = abdomen_dataset.AbdomenDataset(dictionary=dictionary_, question_len=12, mode='val', transform=composed_transforms_ts, return_size=False, seed=args.seed)
    a_trainloader = DataLoader(a_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    a_valloader = DataLoader(a_valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    num_iter_tr = len(a_trainloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.batch_size * nitrs
    print('each_epoch_num_iter: %d' % (num_iter_tr))

    global_step = 0

    a_ques_epoch_losses = []
    a_epoch_losses = []
    epoch_losses = []
    
    a_ques_recent_losses = []
    a_recent_losses = []
    recent_losses = []
    start_t = time.time()
    print('Training Network')

    for epoch in range(args.resume_epoch, args.nepochs):
        
        backbone.train()
        a_decoder.train()
        cmsa.train()
        a_ques_epoch_losses = []
        a_epoch_losses = []
        epoch_losses = []
        

        for ii, a_sample_batched in enumerate(a_trainloader):
            a_img, a_label = a_sample_batched['image'], a_sample_batched['label']
            a_question, a_question_label = a_sample_batched['question'], a_sample_batched['question_label']
        
            a_img, a_label = a_img.cuda(), a_label.cuda()
            a_question, a_question_label = a_question.cuda(), a_question_label.cuda()

            global_step += args.batch_size
            
            a_feats = backbone.forward(a_img)
            a_out = a_decoder.forward(a_feats)
            a_cmsa_out = cmsa.forward(a_feats[-1], a_question)
            # print('a_cmsa_out1.shape: {}'.format(a_cmsa_out.shape))
            a_cmsa_out = cmsa.classify(a_cmsa_out)
            # print('a_cmsa_out2.shape: {}'.format(a_cmsa_out.shape))
            # print('model_out: {}'.format(a_cmsa_out))
            # print('a_question_label: {}'.format(a_question_label))

            a_loss = utils.CELoss(logit=a_out, target=a_label, reduction='mean')
            a_ques_loss = utils.CELoss(logit=a_cmsa_out, target=a_question_label, reduction='mean')

            a_ques_trainloss = a_ques_loss.item()
            a_ques_epoch_losses.append(a_ques_trainloss)
            if len(a_ques_recent_losses) < args.log_every:
                a_ques_recent_losses.append(a_ques_trainloss)
            else:
                a_ques_recent_losses[nitrs % len(a_ques_recent_losses)] = a_ques_trainloss

            a_trainloss = a_loss.item()
            a_epoch_losses.append(a_trainloss)
            if len(a_recent_losses) < args.log_every:
                a_recent_losses.append(a_trainloss)
            else:
                a_recent_losses[nitrs % len(a_recent_losses)] = a_trainloss

            loss = a_loss + a_ques_loss

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss
            
            backbone_optim.zero_grad()
            a_decoder_optim.zero_grad()
            cmsa_optim.zero_grad()

            loss.backward()

            nitrs += 1
            nsamples += args.batch_size

            params = []
            params += filter(lambda p: p.requires_grad, backbone.parameters())
            params += filter(lambda p: p.requires_grad, a_decoder.parameters())
            params += filter(lambda p: p.requires_grad, cmsa.parameters())
            
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(params, args.clip_norm)

            backbone_optim.step()
            a_decoder_optim.step()
            cmsa_optim.step()

            if nitrs % args.log_every == 0:
                a_meanloss = sum(a_recent_losses) / len(a_recent_losses)
                a_ques_meanloss = sum(a_ques_recent_losses) / len(a_ques_recent_losses)
                meanloss = sum(recent_losses) / len(recent_losses)
                
                print('epoch: %d ii: %d a_trainloss: %.2f a_ques_trainloss: %.2f trainloss: %.2f timecost:%.2f secs'%(
                    epoch, ii, a_meanloss, a_ques_meanloss, meanloss, time.time()-start_t))
                writer.add_scalar('data/a_trainloss',a_meanloss,nsamples)
                writer.add_scalar('data/a_ques_trainloss',a_ques_meanloss,nsamples)
                writer.add_scalar('data/a_trainloss',a_meanloss,nsamples)
                
            if ii % (num_iter_tr // 10) == 0:
                grid_image = make_grid(a_img[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('A_Image', grid_image, global_step)                
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.argmax(a_out[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('A_Prediction', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(a_label[:3].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('A_Label', grid_image, global_step)           
            
            # print(wl_params)                

        # validation
        backbone.eval()
        a_decoder.eval()
        cmsa.eval()
        
        a_miou = 0.0
        ques_acc = 0.0

        for ii, a_sample_batched in enumerate(a_valloader):
            a_img, a_label = a_sample_batched['image'], a_sample_batched['label']
            a_question, a_question_label = a_sample_batched['question'], a_sample_batched['question_label']
            a_img, a_label = a_img.cuda(), a_label.cuda()
            a_question, a_question_label = a_question.cuda(), a_question_label.cuda()

            a_feats = backbone.forward(a_img)
            a_out = a_decoder.forward(a_feats)
            a_cmsa_out = cmsa.forward(a_feats[-1], a_question)
            a_cmsa_out = cmsa.classify(a_cmsa_out)

            iou = utils.cal_iou(a_out, a_label, a_classes)
            acc = utils.cal_acc(a_cmsa_out, a_question_label)
            ques_acc += (acc * a_img.shape[0])
            a_miou += iou

        a_miou /= len(a_valset)
        ques_acc /= len(a_valset)

        print('Validation:')
        print('epoch: %d, a_images: %d a_miou: %.4f ques_acc: %.4f' % (
            epoch, len(a_valset), a_miou, ques_acc))
        writer.add_scalar('data/valid_a_miou', a_miou, nsamples)
        writer.add_scalar('data/valid_ques_acc', ques_acc, nsamples)

        if epoch % args.save_every == args.save_every - 1:
            backbone_save_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(epoch) + '.pth')
            a_decoder_save_path = os.path.join(save_dir, 'models', 'a_decoder_epoch-' + str(epoch) + '.pth')
            cmsa_save_path = os.path.join(save_dir, 'models', 'cmsa_epoch-' + str(epoch) + '.pth')
            torch.save(backbone.state_dict(), backbone_save_path)
            torch.save(a_decoder.state_dict(), a_decoder_save_path)
            torch.save(cmsa.state_dict(), cmsa_save_path)
            print("Save backbone at {}\n".format(backbone_save_path))
            print("Save a_decoder at {}\n".format(a_decoder_save_path))
            print("Save cmsa at {}\n".format(cmsa_save_path))
            
        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            adjust_learning_rate(backbone_optim, lr_)
            adjust_learning_rate(a_decoder_optim, lr_)
            adjust_learning_rate(cmsa_optim, lr_)
        



if __name__ == '__main__':
    args = get_arguments()
    main(args)