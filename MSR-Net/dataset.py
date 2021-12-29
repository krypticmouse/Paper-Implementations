import cv2
from config import *
from glob import glob
from torchvision import transforms

class LowLightDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.imgs = [i.split('/')[-1] for i in glob(f'{folder_path}/high/*.png')]

        self.transform = transforms.Compose([
                                             transforms.ToPILImage(),
                                             transforms.Resize((384, 384)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        high_path = f'{self.folder_path}/high/{img}'
        low_path = f'{self.folder_path}/low/{img}'

        high_img = cv2.imread(high_path)
        low_img = cv2.imread(low_path)

        high_img = self.transform(high_img)
        low_img = self.transform(low_img)

        return {
            'high': high_img,
            'low': low_img
        }