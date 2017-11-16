#!/usr/bin/python
# encoding: utf-8

import random
import sys
from glob import glob
from os.path import basename
from os.path import join

import numpy as np
import pandas as pd
# import lmdb
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class hwrDataset(Dataset):
    def __init__(self, root="./data/words/*/*/*.png", mode="train", transform=None, target_transform=None,
                 return_index=False, extra_path=''):
        self.return_index = return_index
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        train_threshold = 0.75

        if return_index:
            assert extra_path != '' and mode == 'test', 'Returns index in run time only.'

            train_threshold = 0.00
            root = join('./data/words/', extra_path, '*.png')
            files = sorted(glob(root))
            print('Files, after order ' + str(files))
        else:
            files = glob(root)

        name_to_file = {}
        rows = []

        for file_name in files:
            name_to_file[basename(file_name).rstrip(".png")] = file_name

        # Reduce the time it takes for this if needed.
        for line in open("./data/words_gt.txt", "r").readlines():
            parts = line.split(" ")
            if parts[0] in name_to_file:
                gt = " ".join(parts[8:]).rstrip("\n")

                # Filters out characters not in the alphabet.
                processed_gt = "".join([k for k in gt.lower() if k in alphabet])
                if len(processed_gt) > 0:
                    rows.append([parts[0], name_to_file[parts[0]], processed_gt])

        if mode == "train":
            self.data = pd.DataFrame(rows[:int(len(rows) * train_threshold)], columns=["name", "path", "groundtruth"])

        else:
            self.data = pd.DataFrame(rows[int(len(rows) * train_threshold):], columns=["name", "path", "groundtruth"])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(self.data.__len__() - 1)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        index += 1

        try:
            img = Image.open(list(self.data.iloc[[index]].path)[0]).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        except Exception:
            print(index)

        if self.transform is not None:
            img = self.transform(img)

        label = str(list(self.data.iloc[[index]].groundtruth)[0])

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_index:
            return img, label, index

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
