"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding, SelfAttention
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading, generate_spatial_batch
from simple_cnn import SimpleCNN
from auto_encoder import Auto_Encoder_Model
from multi_task import ResNet50, ResNet18, ResNet34
from mc import MCNet
from convert import Convert, GAPConvert
import os
from non_local import NONLocalBlock3D
from transformer.SubLayers import MultiHeadAttention


# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                 ae_v_emb, mt_v_emb, distmodal_emb):
        super(BAN_Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.op = args.op
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()
        if args.maml:
            self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, 64)
        if args.multitask:
            self.mt_v_emb = mt_v_emb
            self.mt_convert = nn.Linear(100352, args.mt_feat_dim)
        if args.distmodal:
            self.modal_classifier = distmodal_emb[0]
            self.abd_v_emb = distmodal_emb[1]
            self.abd_convert = distmodal_emb[2]
            self.brain_v_emb = distmodal_emb[3]
            self.brain_convert = distmodal_emb[4]
            self.chest_v_emb = distmodal_emb[5]
            self.chest_convert = distmodal_emb[6]
            self.softmax = nn.Softmax(dim=-1)

        if args.distmodal and args.att_model_path != None:
            ban_pretrained = torch.load(args.att_model_path)
            ban_pretrained_keys = list(ban_pretrained.keys())
            cnt = 0
            w_emb_state_dict = self.w_emb.state_dict()
            q_emb_state_dict = self.q_emb.state_dict()
            v_att_state_dict = self.v_att.state_dict()
            b_net_state_dict = self.b_net.state_dict()
            q_prj_state_dict = self.q_prj.state_dict()

            print('Loading w_emb & q_emb & v_att...')
            for k, v in w_emb_state_dict.items():
                if v.shape == ban_pretrained[ban_pretrained_keys[cnt]].shape:
                    w_emb_state_dict[k] = ban_pretrained[ban_pretrained_keys[cnt]]
                    cnt += 1
            for k, v in q_emb_state_dict.items():
                if v.shape == ban_pretrained[ban_pretrained_keys[cnt]].shape:
                    q_emb_state_dict[k] = ban_pretrained[ban_pretrained_keys[cnt]]
                    cnt += 1
            for k, v in v_att_state_dict.items():
                if v.shape == ban_pretrained[ban_pretrained_keys[cnt]].shape:
                    v_att_state_dict[k] = ban_pretrained[ban_pretrained_keys[cnt]]
                    cnt += 1
            print('Loading b_net & q_prj...')
            for k, v in b_net_state_dict.items():
                if v.shape == ban_pretrained[ban_pretrained_keys[cnt]].shape:
                    b_net_state_dict[k] = ban_pretrained[ban_pretrained_keys[cnt]]
                    cnt += 1
            for k, v in q_prj_state_dict.items():
                if v.shape == ban_pretrained[ban_pretrained_keys[cnt]].shape:
                    q_prj_state_dict[k] = ban_pretrained[ban_pretrained_keys[cnt]]
                    cnt += 1

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.multitask:
            mt_v_emb = self.mt_v_emb(v[2])
            mt_v_emb = mt_v_emb.view(mt_v_emb.shape[0], -1)
            mt_v_emb = self.mt_convert(mt_v_emb).unsqueeze(1)
            v_emb = mt_v_emb
        if self.args.distmodal:
            modal = self.modal_classifier(v[2])
            modal_softmax = self.softmax(modal)
            abd_v_emb = self.abd_v_emb(v[2])
            abd_v_emb = self.abd_convert(abd_v_emb)
            brain_v_emb = self.brain_v_emb(v[2])
            brain_v_emb = self.brain_convert(brain_v_emb)
            chest_v_emb = self.chest_v_emb(v[2])
            chest_v_emb = self.chest_convert(chest_v_emb)
            v_emb = modal_softmax[:, 0].unsqueeze(dim=1) * abd_v_emb + modal_softmax[:, 1].unsqueeze(
                dim=1) * brain_v_emb + modal_softmax[:, 2].unsqueeze(dim=1) * chest_v_emb
            # v_emb = abd_v_emb + brain_v_emb + chest_v_emb
            v_emb = v_emb.unsqueeze(1)

        # print("q.shape: {}".format(q.shape))
        # get lextual feature
        w_emb = self.w_emb(q)
        # print("w_emb.shape: {}".format(w_emb.shape))
        if self.args.self_att:
            q_emb = self.q_emb.forward(w_emb, None)
        else:
            q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]

        # print("q_emb.shape: {}, v_emb.shape: {}".format(q_emb.shape, v_emb.shape))
        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb)  # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:, g, :, :])  # b x l x h
            atten, _ = logits[:, g, :, :].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        if self.args.autoencoder:
            return q_emb.sum(1), decoder
        if self.args.distmodal:
            return q_emb.sum(1), modal
        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)


# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb, mt_v_emb):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        if args.maml:
            self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, 64)
        if args.multitask:
            self.mt_v_emb = mt_v_emb
            # may need to modify
            self.mt_convert = nn.Linear(100352, 1024)  # 128

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.multitask:
            mt_v_emb = self.mt_v_emb(v[2])
            mt_v_emb = mt_v_emb.view(mt_v_emb.shape[0], -1)
            mt_v_emb = self.mt_convert(mt_v_emb).unsqueeze(1)
            v_emb = mt_v_emb
        # get textual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state
        # Attention
        att = self.v_att(v_emb, q_emb)
        if self.args.autoencoder:
            return att, decoder
        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)


# Build BAN model
def build_BAN(dataset, args, priotize_using_counter=False):
    # init word embedding module, question embedding module, and Attention network
    emb_dim = {'glove': 300, 'biowordvec': 200, 'biosentvec': 700}
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim[args.emb_init], .0, args.op)
    if args.self_att:
        print('Self Attention for question embedding')
        d_word_vec = emb_dim[args.emb_init] if 'c' not in args.op else 2 * emb_dim[args.emb_init]
        d_model = d_word_vec
        q_emb = SelfAttention(d_word_vec=d_word_vec, d_model=d_model)
        v_att = BiAttention(dataset.v_dim, d_model, args.num_hid, args.gamma)
        q_dim = d_model
    else:
        q_emb = QuestionEmbedding(emb_dim[args.emb_init] if 'c' not in args.op else 2 * emb_dim[args.emb_init],
                                  args.num_hid, 1, False, .0, args.rnn)
        v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
        q_dim = args.num_hid

    # build and load pre-trained MAML model
    if args.maml:
        weight_path = args.RAD_dir + '/' + args.maml_model_path
        print('load initial weights MAML from: %s' % (weight_path))
        maml_v_emb = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.RAD_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s' % (weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained multi-task model
    if args.multitask:
        mt_v_emb = ResNet50(model_path=args.mt_model_path)
        # weight_path = args.RAD_dir + '/' + args.mt_model_path
        print('load initial weights MT from: %s' % (args.mt_model_path))
        # mt_v_emb.load_state_dict(torch.load(weight_path))
    if args.distmodal:
        modal_classifier = MCNet(input_size=224, in_channels=3, n_classes=3)
        if args.modal_classifier_path is not None:
            modal_classifier.load_state_dict(torch.load(args.modal_classifier_path))
        abd_v_emb = ResNet34(model_path=args.abd_model_path)
        if 'GAP' in args.abd_model_path:
            abd_convert = GAPConvert()
        else:
            abd_convert = Convert(image_size=224, backbone_output_dim=512, os=32, v_dim=args.dm_feat_dim)
            if os.path.exists(args.abd_model_path.replace('backbone', 'convert')) and 'backbone' in args.abd_model_path:
                if args.split == 'train':
                    print('Loading Abdomen Convert Module...')
                    abd_convert.load_state_dict(torch.load(args.abd_model_path.replace('backbone', 'convert')))
        brain_v_emb = ResNet34(model_path=args.brain_model_path)
        if 'GAP' in args.brain_model_path:
            brain_convert = GAPConvert()
        else:
            brain_convert = Convert(image_size=224, backbone_output_dim=512, os=32, v_dim=args.dm_feat_dim)
            if os.path.exists(
                    args.brain_model_path.replace('backbone', 'convert')) and 'backbone' in args.brain_model_path:
                if args.split == 'train':
                    print('Loading Brain Convert Module...')
                    brain_convert.load_state_dict(torch.load(args.brain_model_path.replace('backbone', 'convert')))
        chest_v_emb = ResNet34(model_path=args.chest_model_path)
        if 'GAP' in args.chest_model_path:
            chest_convert = GAPConvert()
        else:
            chest_convert = Convert(image_size=224, backbone_output_dim=512, os=32, v_dim=args.dm_feat_dim)
            if os.path.exists(
                    args.chest_model_path.replace('backbone', 'convert')) and 'backbone' in args.chest_model_path:
                if args.split == 'train':
                    print('Loading Chest Convert Module...')
                    chest_convert.load_state_dict(torch.load(args.chest_model_path.replace('backbone', 'convert')))
        distmodal_emb = [modal_classifier, abd_v_emb, abd_convert, brain_v_emb, brain_convert, chest_v_emb,
                         chest_convert]

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # Optional module: counter for BAN
    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    if use_counter or priotize_using_counter:
        counter = Counter(objects)
    else:
        counter = None
    # init BAN residual network
    b_net = []
    q_prj = []
    c_prj = []
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, q_dim, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, q_dim], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    # init classifier
    classifier = SimpleClassifier(
        q_dim, q_dim * 2, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         ae_v_emb, None, None)
    elif args.maml:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         None, None, None)
    elif args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         ae_v_emb, None, None)
    elif args.multitask:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         None, mt_v_emb, None)
    elif args.distmodal:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         None, None, distmodal_emb)
    return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None, None, None,
                     None)


# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)
    # build and load pre-trained MAML model
    if args.maml:
        weight_path = args.RAD_dir + '/' + args.maml_model_path
        print('load initial weights MAML from: %s' % (weight_path))
        maml_v_emb = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.RAD_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s' % (weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained multi-task model
    if args.multitask:
        mt_v_emb = ResNet50(model_path=args.mt_model_path)
        # weight_path = args.RAD_dir + '/' + args.mt_model_path
        print('load initial weights MT from: %s' % (args.mt_model_path))
        # mt_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb, None)
    elif args.maml:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, None, None)
    elif args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, ae_v_emb, None)
    elif args.multitask:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None, mt_v_emb)
    return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None)


class CMSA_Model(nn.Module):
    def __init__(self, w_emb, q_emb, cmsa, fc, classifier, args, distmodal_emb, cma):
        super(CMSA_Model, self).__init__()
        self.args = args
        self.op = args.op
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.cmsa0 = cmsa[0]
        self.cmsa1 = cmsa[1]
        self.fc = fc
        self.classifier = classifier
        self.cma = cma
        if args.distmodal:
            self.modal_classifier = distmodal_emb[0]
            self.abd_v_emb = distmodal_emb[1]
            self.brain_v_emb = distmodal_emb[2]
            self.chest_v_emb = distmodal_emb[3]
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, v, q):
        """Forward

        v: [batch, 3, h, w]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.distmodal:
            modal = self.modal_classifier(v[2])
            modal_softmax = self.softmax(modal)
            # modal_softmax = self.softmax(modal_softmax)
            abd_v_emb = self.abd_v_emb(v[2])
            brain_v_emb = self.brain_v_emb(v[2])
            chest_v_emb = self.chest_v_emb(v[2])
            # v_emb = (abd_v_emb + brain_v_emb + chest_v_emb)/3
            v_emb = modal_softmax[:, 0].view(modal_softmax.shape[0], 1, 1, 1) * abd_v_emb + modal_softmax[:, 1].view(modal_softmax.shape[0], 1, 1, 1) * brain_v_emb + modal_softmax[:, 2].view(modal_softmax.shape[0], 1, 1,1) * chest_v_emb
        else:
            raise NotImplementedError

        b, c, h, w = v_emb.shape
        spatial = generate_spatial_batch(b, h, w)
        spatial = torch.from_numpy(spatial).to(v_emb.get_device())

        # get lextual feature
        w_emb = self.w_emb(q)
        if self.args.self_att:
            q_emb = self.q_emb.forward(w_emb, None)
        else:
            q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]

        feat_cat_lst = []
        for i in range(q_emb.shape[1]):
            lang_feat = q_emb[:, i, :].reshape((q_emb.shape[0], q_emb.shape[2], 1, 1))
            lang_feat = lang_feat.repeat((1, 1, h, w))
            if self.args.use_spatial:
                feat_cat = torch.cat((v_emb, lang_feat, spatial), dim=1)
            else:
                feat_cat = torch.cat((v_emb, lang_feat), dim=1)
            feat_cat_lst.append(feat_cat)
        cm_feat = torch.cat([feat_cat.unsqueeze(dim=2) for feat_cat in feat_cat_lst],
                            dim=2)  # b x c x q_len x h x w (c=v_dim + q_dim + 8)

        cm_feat1 = self.cmsa0(cm_feat)
        cm_feat1 = cm_feat1+cm_feat
        cm_feat = self.cmsa1(cm_feat1)
        cm_feat = cm_feat.view(cm_feat.shape[0], cm_feat.shape[1], cm_feat.shape[2], -1)
        cm_feat = torch.mean(cm_feat, dim=-1)
        cm_feat = cm_feat.permute(0, 2, 1)
        cm_feat = self.fc(cm_feat)

        if self.cma is not None:
            q_emb, _ = self.cma(q=q_emb, k=cm_feat, v=cm_feat)
        else:
            q_emb = q_emb + cm_feat

        if self.args.distmodal:
            return q_emb.sum(1), modal

        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)


def build_CMSA(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    emb_dim = {'glove': 300, 'biowordvec': 200, 'biosentvec': 700}
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim[args.emb_init], .0, args.op)
    w_dim = emb_dim[args.emb_init] if 'c' not in args.op else 2 * emb_dim[args.emb_init]
    if args.self_att:
        print('Use Self Attention as question embedding')
        q_dim = w_dim
        q_emb = SelfAttention(d_word_vec=w_dim, d_model=q_dim)
    else:
        print('Use LSTM as question embedding')
        q_dim = args.num_hid
        q_emb = QuestionEmbedding(w_dim, q_dim, 1, False, .0, args.rnn)

    if args.distmodal:
        modal_classifier = MCNet(input_size=224, in_channels=3, n_classes=3)
        if args.modal_classifier_path is not None:
            modal_classifier.load_state_dict(torch.load(args.modal_classifier_path))
        abd_v_emb = ResNet34(model_path=args.abd_model_path)
        brain_v_emb = ResNet34(model_path=args.brain_model_path)
        chest_v_emb = ResNet34(model_path=args.chest_model_path)
        distmodal_emb = [modal_classifier, abd_v_emb, brain_v_emb, chest_v_emb]
        v_dim = 512
    else:
        raise NotImplementedError

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    if args.use_spatial:
        cmsa0 = NONLocalBlock3D(in_channels=v_dim + q_dim + 8, inter_channels=None, sub_sample=False, bn_layer=True)
        cmsa1 = NONLocalBlock3D(in_channels=v_dim + q_dim + 8, inter_channels=None, sub_sample=False, bn_layer=True)
        fc = nn.Linear(v_dim + q_dim + 8, q_dim)
    else:
        cmsa = NONLocalBlock3D(in_channels=v_dim + q_dim, inter_channels=None, sub_sample=False, bn_layer=True)
        fc = nn.Linear(v_dim + q_dim, q_dim)

    if args.distmodal and args.att_model_path != None:
        att_pretrained = torch.load(args.att_model_path)
        att_pretrained_keys = list(att_pretrained.keys())
        cnt = 0
        w_emb_state_dict = w_emb.state_dict()
        q_emb_state_dict = q_emb.state_dict()
        cmsa0_state_dict = cmsa0.state_dict()
        cmsa1_state_dict = cmsa1.state_dict()
        fc_state_dict = fc.state_dict()

        print('Loading w_emb & q_emb & cmsa & fc...')
        for k, v in w_emb_state_dict.items():
            if v.shape == att_pretrained[att_pretrained_keys[cnt]].shape:
                w_emb_state_dict[k] = att_pretrained[att_pretrained_keys[cnt]]
                cnt += 1
        for k, v in q_emb_state_dict.items():
            if v.shape == att_pretrained[att_pretrained_keys[cnt]].shape:
                q_emb_state_dict[k] = att_pretrained[att_pretrained_keys[cnt]]
                cnt += 1
        for k, v in cmsa0_state_dict.items():
            if v.shape == att_pretrained[att_pretrained_keys[cnt]].shape:
                cmsa0_state_dict[k] = att_pretrained[att_pretrained_keys[cnt]]
                cnt += 1
        for k, v in cmsa1_state_dict.items():
            if v.shape == att_pretrained[att_pretrained_keys[cnt]].shape:
                cmsa1_state_dict[k] = att_pretrained[att_pretrained_keys[cnt]]
                cnt += 1
        for k, v in fc_state_dict.items():
            if v.shape == att_pretrained[att_pretrained_keys[cnt]].shape:
                fc_state_dict[k] = att_pretrained[att_pretrained_keys[cnt]]
                cnt += 1
    cmsa = [cmsa0, cmsa1]

    classifier = SimpleClassifier(
        q_dim, q_dim * 2, dataset.num_ans_candidates, args)

    cma = None
    if args.use_cma:
        cma = MultiHeadAttention(n_head=1, d_model=q_dim, d_k=q_dim, d_v=q_dim)

    return CMSA_Model(w_emb, q_emb, cmsa, fc, classifier, args, distmodal_emb, cma)
