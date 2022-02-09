import os
import os.path as osp
# import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class tieredImageNet(Dataset):

    def __init__(self, setname, args=None):
        TRAIN_PATH = osp.join(args.data_dir, 'tiered_imagenet-deep/train')
        VAL_PATH = osp.join(args.data_dir, 'tiered_imagenet-deep/val')
        TEST_PATH = osp.join(args.data_dir, 'tiered_imagenet-deep/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')

        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path

        fine_folders = [os.path.join(coarse_label, label) \
                        for coarse_label in coarse_folders \
                        if os.path.isdir(coarse_label) \
                        for label in os.listdir(coarse_label)
                        ]
        coarse_labels = np.array(range(len(coarse_folders)))
        coarse_labels = dict(zip(coarse_folders, coarse_labels))
        # 细类标签
        labels = np.array(range(len(fine_folders)))
        labels = dict(zip(fine_folders, labels))

        data = []
        for c in fine_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            data += temp

        # 粗类标签
        coarse_label = [coarse_labels['/' + self.get_coarse_class(x)] for x in data]

        # 细类标签
        fine_label = [labels['/' + self.get_class(x)] for x in data]

        self.data = data
        self.coarse_label = coarse_label
        self.label = fine_label
        self.num_fine_class = len(set(fine_label))
        self.num_coarse_class = len(set(coarse_label))

        # Transformation
        if setname == 'val' or setname == 'test':
            image_size = 84
            resize_size = 92
            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def get_coarse_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, coarse_label, label = self.data[i], self.coarse_label[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, coarse_label


if __name__ == '__main__':
    pass