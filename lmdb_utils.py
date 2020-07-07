import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import lmdb
from caffe_pb2 import Datum
import random

def transform(image):
    img_size = image.shape[0]
    # resizing 
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))  
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1) 
    # rotation
    if random.random() > 0.75:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # grayscale conversion
    if random.random() > 0.75:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image


class SingleLMDBDataset(Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, source_lmdb, source_filelists, key=None):
        self.env = lmdb.open(source_lmdb, max_readers=4, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.train_list = []
        for line in source_filelists:
            l = line.rstrip().lstrip()
            if len(l) > 0:
                groups = l.split(' ')
                lmdb_key1 = groups[0]
                lmdb_key2 = groups[1]
                label = groups[2]
                self.train_list.append([lmdb_key1, lmdb_key2,int(label)])
        self.key = key

    def close(self):
        self.txn.abort()
        self.env.close()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        if index > len(self.train_list):
            raise IndexError("index exceeds the max-length of the dataset.")
        lmdb_key1, lmdb_key2, label = self.train_list[index]
        datum = Datum()
        real_byte_buffer = self.txn.get(lmdb_key1.encode('utf-8'))
        raw_real_byte_buffer2 = self.txn.get(lmdb_key2.encode('utf-8'))
        datum.ParseFromString(real_byte_buffer)
        image1 = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)
        datum.ParseFromString(raw_real_byte_buffer2)
        image2 = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)

        image1 = transform(image1)
        image2 = transform(image2)
        img = torch.cat([image1,image2], dim=0)

        return img, label

        
        
