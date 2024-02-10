import torch
from utils import *
import numpy as np
import os

class CT_dataset(torch.utils.data.Dataset):

    def __init__(self, directory, unet = False):

        self.directory = directory

        self.name_list = sorted(os.listdir(self.directory))
        self.unet = unet


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        file = np.load(os.path.join(self.directory,file_name))
        image = file['image']
        image = torch.tensor(image, dtype = torch.float32)


        if self.unet:

            fbp = file['fbp']
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[None,...], fbp
        
        else:
            sinogram = file['sinogram']
            sinogram = torch.tensor(sinogram, dtype = torch.float32)

            image = image.reshape(-1, 1)
            return image, sinogram
        
    
    


class CT_odl(torch.utils.data.Dataset):

    def __init__(self, directory):

        self.directory = directory
        self.name_list = sorted(os.listdir(self.directory))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        data = np.load(os.path.join(self.directory,file_name))
        image = data['image']
        sinogram = data['sinogram']
        fbp = data['fbp']

        image = torch.tensor(image, dtype = torch.float32)[None,...]
        fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
        sinogram = torch.tensor(sinogram, dtype = torch.float32)[None,...]
            
        return image, sinogram, fbp

    
