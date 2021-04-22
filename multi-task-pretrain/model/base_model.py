import torch
import torch.nn as nn
from model.attention import BiAttention, StackedAttention
from model.language_model import WordEmbedding, QuestionEmbedding, SelfAttention
from model.classifier import SimpleClassifier
from model.fc import FCNet
from model.bc import BCNet
from dataloaders import dictionary
import os
import numpy as np

import fasttext
from model.non_local import NONLocalBlock3D

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, b_net, q_prj, classifier, args):
        super(BAN_Model, self).__init__()
        self.args = args
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.classifier = classifier

    def forward(self, v, q):
        '''
        v: visual feature [batch_size, v_dim]
        q: question words [batch_size, seq_length]
        '''
        assert len(v.shape) == 2
        assert v.shape[-1] == self.args.v_dim

        v_emb = v.unsqueeze(1)
        
        # get lextual feature
        w_emb = self.w_emb(q)
        if self.args.self_att:
            q_emb = self.q_emb.forward(w_emb, None)
        else:
            q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)

def tfidf_loading(use_tfidf, w_emb, args):
    if use_tfidf:
        dict = dictionary.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))

        if args.emb_init == 'glove':
            # load extracted tfidf and weights from file for saving loading time
            if os.path.isfile(os.path.join(args.RAD_dir, 'embed_tfidf_weights.pkl')) == True:
                print("Loading embedding tfidf and weights from file")
                with open(os.path.join(args.RAD_dir ,'embed_tfidf_weights.pkl'), 'rb') as f:
                    w_emb = torch.load(f)
                print("Load embedding tfidf and weights from file successfully")
            else:
                print("Embedding tfidf and weights haven't been saving before")
                tfidf, weights = tfidf_from_questions(['train'], None, dict)
                w_emb.init_embedding(os.path.join(args.RAD_dir, 'glove6b_init_300d.npy'), tfidf, weights)
                with open(os.path.join(args.RAD_dir ,'embed_tfidf_weights.pkl'), 'wb') as f:
                    torch.save(w_emb, f)
                print("Saving embedding with tfidf and weights successfully")
        elif args.emb_init == 'biowordvec':
            if os.path.isfile(os.path.join(args.RAD_dir, 'embed_tfidf_weights_biowordvec.pkl')) == True:
                print("Loading embedding tfidf and weights (biowordvec) from file")
                with open(os.path.join(args.RAD_dir ,'embed_tfidf_weights_biowordvec.pkl'), 'rb') as f:
                    w_emb = torch.load(f)
                print("Load embedding tfidf and weights (biowordvec) from file successfully")
            else:
                print("Embedding tfidf and weights (biowordvec) haven't been saving before")
                tfidf, weights = tfidf_from_questions(['train'], args, dict)
                w_emb.init_embedding(os.path.join(args.RAD_dir, 'biowordvec_init_200d.npy'), tfidf, weights)
                with open(os.path.join(args.RAD_dir ,'embed_tfidf_weights_biowordvec.pkl'), 'wb') as f:
                    torch.save(w_emb, f)
                print("Saving embedding with tfidf and weights (biowordvec) successfully")
    return w_emb

def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    if args.use_RAD:
        dataroot = args.RAD_dir
    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    if args.emb_init == 'glove':
        # Latent word embeddings
        emb_dim = 300
        glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
        weights, word2emb = create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    elif args.emb_init == 'biowordvec':
        bio_bin_file = '/data1/chenguanqi/Medical-VQA/download_file/BioWordVec_PubMed_MIMICIII_d200.bin'
        weights = create_biowordvec_embedding_init(dictionary.idx2word[N:], bio_bin_file)
    else:
        weights = None
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

def create_biowordvec_embedding_init(idx2word, bio_bin_file):
    emb_dim = 200
    model = fasttext.load_model(bio_bin_file)
    print('model successfully loaded')
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)
    for idx, word in enumerate(idx2word):
        weights[idx] = model.get_word_vector(word)
    return weights

# Build BAN model
def build_BAN(dictionary, args):
    # init word embedding module, question embedding module, and Attention network
    emb_dim = {'glove':300, 'biowordvec':200, 'biosentvec':700}
    w_emb = WordEmbedding(dictionary.ntoken, emb_dim[args.emb_init], .0, args.op)
    w_dim = emb_dim[args.emb_init] if 'c' not in args.op else 2*emb_dim[args.emb_init]
    if args.self_att:
        print('Use Self Attention as question embedding')
        q_dim = w_dim
        q_emb = SelfAttention(d_word_vec=w_dim, d_model=q_dim)
        v_att = BiAttention(args.v_dim, q_dim, args.num_hid, args.gamma)
    else:
        print('Use LSTM as question embedding')
        q_dim = args.num_hid
        q_emb = QuestionEmbedding(w_dim, q_dim, 1, False, .0,  args.rnn)
        v_att = BiAttention(args.v_dim, q_dim, args.num_hid, args.gamma)

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    
    # init BAN residual network
    b_net = []
    q_prj = []
    for i in range(args.gamma):
        b_net.append(BCNet(args.v_dim, q_dim, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, q_dim], '', .2))
        
    # init classifier
    classifier = SimpleClassifier(
        q_dim, q_dim * 2, 2, args)

    return BAN_Model(w_emb, q_emb, v_att, b_net, q_prj, classifier, args)


# Create CMSA model
class CMSA_Model(nn.Module):
    def __init__(self, w_emb, q_emb, cmsa, fc, classifier, args):
        super(CMSA_Model, self).__init__()
        self.args = args
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.cmsa = cmsa
        self.fc = fc
        self.classifier = classifier

    def forward(self, v, q):
        '''
        v: visual feature [batch_size, v_dim, h, w]
        q: question words [batch_size, seq_length]
        '''
        assert len(v.shape) == 4
        b,c,h,w = v.shape
        # print('v.shape: {}'.format(v.shape))

        spatial = generate_spatial_batch(b,h,w)
        spatial = torch.from_numpy(spatial).to(v.get_device())
        # print('spatial.shape: {}'.format(spatial.shape))
        
        # get lextual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # print('q_emb.shape: {}'.format(q_emb.shape))

        feat_cat_lst = []
        for i in range(q_emb.shape[1]):
            lang_feat = q_emb[:,i,:].reshape((q_emb.shape[0], q_emb.shape[2], 1, 1))
            lang_feat = lang_feat.repeat((1, 1, h, w))
            if self.args.use_spatial:
                feat_cat = torch.cat((v, lang_feat, spatial), dim=1)
            else:
                feat_cat = torch.cat((v, lang_feat), dim=1)
            feat_cat_lst.append(feat_cat)
        cm_feat = torch.cat([feat_cat.unsqueeze(dim=2) for feat_cat in feat_cat_lst], dim=2) # b x c x q_len x h x w (c=v_dim + q_dim + 8)
        # print('cm_feat.shape: {}'.format(cm_feat.shape))

        cm_feat = self.cmsa(cm_feat)
        # print('cmsa_out.shape: {}'.format(cm_feat.shape))
        cm_feat = cm_feat.view(cm_feat.shape[0], cm_feat.shape[1], cm_feat.shape[2], -1)
        cm_feat = torch.mean(cm_feat, dim=-1)
        cm_feat = cm_feat.permute(0, 2, 1)
        cm_feat = self.fc(cm_feat)
        # print('fc_out.shape: {}'.format(cm_feat.shape))
        q_emb = q_emb + cm_feat

        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)

def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, 8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, :, h, w] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val

# Build CMSA model
def build_CMSA(dictionary, args):
    # init word embedding module, question embedding module, and Attention network
    emb_dim = {'glove':300, 'biowordvec':200, 'biosentvec':700}
    w_emb = WordEmbedding(dictionary.ntoken, emb_dim[args.emb_init], .0, args.op)
    w_dim = emb_dim[args.emb_init] if 'c' not in args.op else 2*emb_dim[args.emb_init]
    if args.self_att:
        print('Use Self Attention as question embedding')
        q_dim = w_dim
        q_emb = SelfAttention(d_word_vec=w_dim, d_model=q_dim)  
    else:
        print('Use LSTM as question embedding')
        q_dim = args.num_hid
        q_emb = QuestionEmbedding(w_dim, q_dim, 1, False, .0,  args.rnn)

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    
    if args.use_spatial:
        cmsa = NONLocalBlock3D(in_channels=args.v_dim+q_dim+8, inter_channels=None, sub_sample=False, bn_layer=True)
        fc = nn.Linear(args.v_dim+q_dim+8, q_dim)
    else:
        cmsa = NONLocalBlock3D(in_channels=args.v_dim+q_dim, inter_channels=None, sub_sample=False, bn_layer=True)
        fc = nn.Linear(args.v_dim+q_dim, q_dim)

    # init classifier
    classifier = SimpleClassifier(
        q_dim, q_dim * 2, 2, args)

    return CMSA_Model(w_emb, q_emb, cmsa, fc, classifier, args)