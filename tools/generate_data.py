import numpy as np 
import PIL.Image as Image
import _pickle as cPickle
import json
import os
import pickle

data = cPickle.load(open('images224x224.pkl', 'rb'))
print(data.shape)
# png_202 = data[202]
# png_202 = (png_202 * 255).astype(np.uint8).squeeze()
# print(png_202.shape)
# png_202 = Image.fromarray(png_202)
# png_202.show()

'''
size = 224
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
img_id2idx = json.load(open('imgid2idx.json'))
# a = "synpic29795.jpg"
# print(img_id2idx[a])
# print(img_id2idx.keys())
data_dir = './images'
img_names = list(img_id2idx.keys())

final_np = np.zeros((315, 3, size, size), dtype=np.float32)

for img_name in img_names:
    img_path = os.path.join(data_dir, img_name)
    img = Image.open(img_path)
    img = img.resize((size, size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32)
    img_np /= 255.0
    if len(img_np.shape) == 2:
        img_np = np.stack((img_np, img_np, img_np), axis=-1)
        print(img_np.shape)
    img_np -= mean
    img_np /= std
    img_np = img_np.astype(np.float32).transpose((2, 0, 1))
    final_np[img_id2idx[img_name]] = img_np

with open('images224x224.pkl', 'wb') as f:
    pickle.dump(final_np, f)
'''
