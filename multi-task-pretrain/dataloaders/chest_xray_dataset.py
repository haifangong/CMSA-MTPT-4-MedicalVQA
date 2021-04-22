import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np 
import re
import json

# for image: (224, 224) -> (224, 224, 3)

# label setting:
# 0 for NORMAL, 1 for PNEUMONIA

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

class ChestXrayDataset(data.Dataset):
    def __init__(self, dictionary, question_len=12, mode='train', transform=None, return_size=False, seed=1234):
        self.dictionary = dictionary
        self.transform = transform
        self.return_size = return_size
        self.seed = seed
        np.random.seed(self.seed)

        self.data_dir = '/data1/chenguanqi/Medical-VQA/chest_xray'
        self.trainset_path = '/data1/chenguanqi/Medical-VQA/multi-task-ABC/data/chest_xray_trainset.txt'
        self.valset_path = '/data1/chenguanqi/Medical-VQA/multi-task-ABC/data/chest_xray_valset.txt'

        self.trainjson_path = '/data1/chenguanqi/Medical-VQA/multi-task-modal/data/chest_train.json'
        self.testjson_path = '/data1/chenguanqi/Medical-VQA/multi-task-modal/data/chest_test.json'
        if mode == 'train':
            json_path = self.trainjson_path
        else:
            json_path = self.testjson_path
        self.entries = json.load(open(json_path))
        self.tokenize(question_len)

        self.img_lst = list()
        self.label_lst = list()
        if mode == 'train':
            file_path = self.trainset_path
        else:
            file_path = self.valset_path
        file = open(file_path)
        for line in file:
            img_name, lbl = line.strip("\n").split(" ")
            self.img_lst.append(img_name)
            self.label_lst.append(lbl)

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.img_lst[item])
        lbl = int(self.label_lst[item])

        assert os.path.exists(img_path), ('{} does not exist'.format(img_path))
        
        img = Image.open(img_path)
        w,h = img.size 
        size = (h,w)

        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np, img_np, img_np), axis=-1)
            img = Image.fromarray(img_np.astype(np.uint8))

        label = Image.fromarray(np.zeros(size, dtype=np.uint8))

        sample_t = {'image': img, 'label': label}
        if self.transform:
            sample_t = self.transform(sample_t) 

        sample = {'image': sample_t['image'], 'label':lbl}

        pos_lst = [i.start() for i in re.finditer('/', img_path)]
        label_name = img_path[pos_lst[-2]+1:-4]
        sample['label_name'] = label_name

        arr = np.random.randint(len(self.entries), size=1)
        index = arr[0]
        entry = self.entries[index]
        question = entry['q_token']
        question = torch.from_numpy(np.array(entry['q_token']))
        ques_label = entry['label']
        sample['question'] = question
        sample['question_label'] = ques_label

        return sample

    def __len__(self):
        return len(self.img_lst)
