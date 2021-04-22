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
from model.convert import Convert
from model.base_model import build_BAN

import utils

from dataloaders import dictionary
import numpy as np


def softmax(x):
    # return np.exp(x) / sum(np.exp(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet34')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-dictionary_path', type=str, default='/data1/chenguanqi/Medical-VQA/MICCAI19-MedVQA/data_RAD/dictionary.pkl')
    parser.add_argument('-RAD_dir', type=str, default='/data1/chenguanqi/Medical-VQA/MICCAI19-MedVQA/data_RAD')
    parser.add_argument('-v_dim', type=int, default=1024)
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
    parser.add_argument('-resume_epoch', type=int, default=100)
    parser.add_argument('-train_fold', type=str, default='BrainTumor_ClipGrad-resnet34-biowordvec-SelfAtt')
    parser.add_argument('-run_id', type=int, default=-1)
    parser.add_argument('-T', type=int, default=2)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=20)

    parser.add_argument('-save_every', type=int, default=5)
    parser.add_argument('-log_every', type=int, default=25)
    parser.add_argument('-emb_init', type=str, default='biowordvec', choices=['glove', 'biowordvec', 'biosentvec'])
    parser.add_argument('-self_att', action='store_true', default=False, help='Use Self Attention?')
    parser.add_argument('-result_fold', type=str, default='results')


    return parser.parse_args()

def adjust_learning_rate(optimizer, lr_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

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

    b_decoder = Classifier(in_channels=filters[-1], n_classes=b_classes)
    convert = Convert(image_size=args.input_size, backbone_output_dim=filters[-1], os=32, v_dim=args.v_dim)
    ban = build_BAN(dictionary=dictionary_, args=args)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', args.train_fold, 'run_'+str(run_id))
    
    backbone_resume_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(args.resume_epoch - 1) + '.pth')
    b_decoder_resume_path = os.path.join(save_dir, 'models', 'b_decoder_epoch-' + str(args.resume_epoch - 1) + '.pth')
    convert_resume_path = os.path.join(save_dir, 'models', 'convert_epoch-' + str(args.resume_epoch - 1) + '.pth')
    ban_resume_path = os.path.join(save_dir, 'models', 'ban_epoch-' + str(args.resume_epoch - 1) + '.pth')
    print('Initializing weights from: {}, epoch: {}...'.format(save_dir, args.resume_epoch))
    backbone.load_state_dict(torch.load(backbone_resume_path, map_location=lambda storage, loc: storage))
    b_decoder.load_state_dict(torch.load(b_decoder_resume_path, map_location=lambda storage, loc: storage))
    convert.load_state_dict(torch.load(convert_resume_path, map_location=lambda storage, loc: storage))
    ban.load_state_dict(torch.load(ban_resume_path, map_location=lambda storage, loc: storage))

    torch.cuda.set_device(device=0)
    backbone.cuda()
    b_decoder.cuda()
    convert.cuda()
    ban.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    b_valset = brain_tumor_dataset.BrainTumorDataset(dictionary=dictionary_, question_len=12, task='classification', mode='val', transform=composed_transforms_ts, return_size=False, seed=args.seed)
    b_valloader = DataLoader(b_valset, batch_size=1, shuffle=False, num_workers=2)

    # validation
    backbone.eval()
    b_decoder.eval()
    convert.eval()
    ban.eval()
    b_acc = 0.0
    ques_acc = 0.0

    for ii, b_sample_batched in enumerate(b_valloader):
        b_img, b_label = b_sample_batched['image'], b_sample_batched['label']
        b_img, b_label = b_img.cuda(), b_label.cuda()
        b_question, b_question_label = b_sample_batched['question'], b_sample_batched['question_label']
        b_question, b_question_label = b_question.cuda(), b_question_label.cuda()

        b_feats = backbone.forward(b_img)
        b_outs = b_decoder.forward(b_feats[-1])

        b_convert = convert.forward(b_feats[-1])
        b_ban_out = ban.forward(b_convert, b_question)
        b_ban_out = ban.classify(b_ban_out)

        acc = utils.cal_acc(b_outs, b_label)
        b_acc += acc

        acc = utils.cal_acc(b_ban_out, b_question_label)
        ques_acc += acc
    b_acc /= len(b_valset)
    ques_acc /= len(b_valset)

    print('Validation:')
    print('epoch: %d, b_images: %d b_acc: %.4f ques_acc: %.4f' % (args.resume_epoch, len(b_valset), b_acc, ques_acc))

    result_dir = os.path.join(save_dir_root, args.result_fold, args.train_fold, 'run_'+str(args.run_id))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = open(os.path.join(result_dir, 'result.txt'), 'w')
    result_file.write('epoch: %d, b_images: %d b_acc: %.4f ques_acc: %.4f' % (args.resume_epoch, len(b_valset), b_acc, ques_acc))



if __name__ == '__main__':
    args = get_arguments()
    main(args)