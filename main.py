"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa)
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import dataset_RAD
import base_model
from train import train
import utils

try:
    import _pickle as pickle
except:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/CMSA',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=70,
                        help='the number of epoches') # 20
    parser.add_argument('--lr', default=0.005, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Choices of attention models
    parser.add_argument('--model', type=str, default='CMSA', choices=['BAN', 'SAN', 'CMSA'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=True,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', type=str, default='./data_RAD',
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')
    parser.add_argument('--mt_feat_dim', default=128, type=int,
                        help='visual feature dim')
    parser.add_argument('--dm_feat_dim', default=1024, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the ae_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    # MT component hyper-parameters
    parser.add_argument('--multitask', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--mt_model_path', type=str, default='/data1/chenguanqi/Medical-VQA/multi-task-ABC/run/Baseline_Abdomen_BrainTumor_ChestXray/run_2/models/backbone_epoch-299.pth',
                        help='the mt_model_path we use')

    # three branches hyper-parameters
    parser.add_argument('--distmodal', action='store_true', default=True,
                        help='End to end model?')  # distinguish modal
    # parser.add_argument('--abd_model_path', type=str, default='saved_models/best/CMSA-Abdomen_ClipGrad-resnet34-biowordvec/backbone_epoch-49.pth',
    #                     help='the abd_model_path we use')
    # parser.add_argument('--brain_model_path', type=str, default='saved_models/best/CMSA-BrainTumor_ClipGrad-resnet34-biowordvec/backbone_epoch-99.pth',
    #                     help='the brain_model_path we use')
    # parser.add_argument('--chest_model_path', type=str, default='saved_models/best/CMSA-ChestXray_ClipGrad-resnet34-biowordvec/backbone_epoch-99.pth',
    #                     help='the chest_model_path we use')
    # parser.add_argument('--abd_model_path', type=str, default='saved_models/best/STPT/a.pth',
    #                     help='the abd_model_path we use')
    # parser.add_argument('--brain_model_path', type=str, default='saved_models/best/STPT/b.pth',
    #                     help='the brain_model_path we use')
    # parser.add_argument('--chest_model_path', type=str, default='saved_models/best/STPT/c.pth',
    #                     help='the chest_model_path we use')
    parser.add_argument('--abd_model_path', type=str, default='saved_models/best/resnet34-333f7ec4.pth',
                        help='the abd_model_path we use')
    parser.add_argument('--brain_model_path', type=str, default='saved_models/best/resnet34-333f7ec4.pth',
                        help='the brain_model_path we use')
    parser.add_argument('--chest_model_path', type=str, default='saved_models/best/resnet34-333f7ec4.pth',
                        help='the chest_model_path we use')
    parser.add_argument('--modal_classifier_path', type=str, default='saved_models/best/MCNet/mcnet_epoch-19.pth',
                        help='the modal_classifier_path we use')
    parser.add_argument('--att_model_path', type=str, default='saved_models/best/CMSA-BrainTumor_ClipGrad-resnet34-biowordvec/backbone_epoch-99.pth')

    parser.add_argument('--modal_alpha', type=float, default=0.5)
    parser.add_argument('--emb_init', type=str, default='biowordvec', choices=['glove', 'biowordvec', 'biosentvec'])
    parser.add_argument('--dwa', action='store_true', default=False, help='Use DWA?')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--self_att', action='store_true', default=False, help='Use Self Attention?')
    parser.add_argument('--use_cma', action='store_true', default=False, help='Use CMA?')
    parser.add_argument('--use_spatial', action='store_true', default=False, help='Use spatial feature?')

    # Return args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # create output directory and log file
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    # Set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Load dictionary and RAD training dataset
    if args.use_RAD:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary, question_len=args.question_len)

    batch_size = args.batch_size
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        if args.self_att:
            transformer_layers = torch.nn.ModuleList([model.w_emb, model.q_emb])
            transformer_layers_params = list(map(id, transformer_layers.parameters()))
            other_params = filter(lambda p: id(p) not in transformer_layers_params, model.parameters())
            optim = torch.optim.Adamax([{'params': other_params, 'lr': args.lr},
                                        {'params': transformer_layers.parameters(), 'lr': args.lr * 0.2}])
        else:
            optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1
    # create training dataloader
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
    eval_loader = DataLoader(eval_dset, 1, shuffle=False, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
    # training phase
    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
