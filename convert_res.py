import numpy as np
from train_dataset.toolkit.utils import mask2rle
import sys
import pickle
import os
import cv2

testroot="test_dataset_A"

with open(sys.argv[1], "rb") as f:
    res = pickle.load(f)

with open(os.path.join(testroot, "test.txt"), "r") as f:
    ids = f.readlines()
    ids = list(map(lambda x: x.strip(), ids))

assert len(ids) == len(res)

os.makedirs('visualize', exist_ok=True)
csv = open(r'predict.csv', 'w', encoding='utf-8')
csv.writelines("filename,w h,rle编码\n")

for fn, mask in zip(ids, res):
    img = cv2.imread(os.path.join(testroot, "img", fn))
    h, w = img.shape[:2]
    mask = mask.reshape((h, w)).astype(np.uint8)
    fuse = img.copy()
    fuse[:,:,2] = img[:,:,2].astype(np.float32)*(1-mask.astype(np.float32)) + mask.astype(np.float32)*0.8*255
    cv2.imwrite(os.path.join("visualize", fn), fuse)
    rle = mask2rle(mask)
    csv.writelines("{},{} {},{}\n".format(fn, w, h, rle))

csv.close()