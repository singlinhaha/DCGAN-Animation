import os
import cv2
from torch.utils.data import Dataset
import numpy as np


class DataReader(Dataset):
    def __init__(self, root_path, transforms=None):
        super(DataReader, self).__init__()
        self.transforms = transforms
        self.sample = [os.path.join(root_path, i) for i in os.listdir(root_path)]

    def __getitem__(self, index):
        path = self.sample[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            trans = self.transforms(image=img)
            img = trans['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img

    def __len__(self):
        return len(self.sample)