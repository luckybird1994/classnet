import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

########################### Data Augmentation ###########################
class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0)
        return image, mask

class DatasetTrain(Dataset):
    def __init__(self):

        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

        self.normalize  = Normalize( mean=self.mean, std=self.std )
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.totensor   = ToTensor()

        self.img_dir = "data1/train/imgs/"
        self.label_dir = "data1/train/gts/"
        self.examples = []

        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            if file_name.find(".jpg") != -1:

                img_path = self.img_dir + file_name
                label_img_path = self.label_dir + file_name.replace(".jpg", ".png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["label_name"] = file_name.replace(".jpg", ".png")

                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, idx):

        example = self.examples[idx]

        img_path = example["img_path"]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32)

        label_img_path = example["label_img_path"]
        mask  = cv2.imread(label_img_path, 0).astype(np.float32)

        image, mask = self.normalize(image, mask)
        image, mask = self.randomcrop(image, mask)
        image, mask = self.randomflip(image, mask)
        return image, mask

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask

    def __len__(self):
        return self.num_examples

class DatasetVal(Dataset):
    def __init__(self):

        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

        self.normalize  = Normalize( mean=self.mean, std=self.std )
        self.resize = Resize(352, 352)
        self.totensor   = ToTensor()

        self.img_dir = "data/val/imgs/"
        self.label_dir = "data/val/gts/"
        self.examples = []

        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            if file_name.find(".jpg") != -1:
                img_path = self.img_dir + file_name
                label_img_path = self.label_dir + file_name.replace(".jpg", ".png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["label_name"] = file_name.replace(".jpg", ".png")

                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, idx):

        example = self.examples[idx]

        img_path = example["img_path"]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32)

        label_img_path = example["label_img_path"]
        mask  = cv2.imread(label_img_path, 0).astype(np.float32)

        image, mask = self.normalize(image, mask)
        image, mask = self.resize(image, mask)
        image, mask = self.totensor(image, mask)
        return image, mask

    def __len__(self):
        return self.num_examples

if __name__ == '__main__':

    train_dataset = DatasetTrain()
    val_dataset = DatasetVal()

    train_loader = DataLoader(dataset=train_dataset, collate_fn=train_dataset.collate, batch_size=32, shuffle=True,num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=8)

    num = 0
    for i, data in enumerate(val_loader, 0):
        print( data[1].shape )
        if i >= 10:
            exit()
