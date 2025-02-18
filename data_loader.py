import torch
from utils import *
import numpy as np
import os
import imageio
import torch.nn.functional as F
from skimage.transform import iradon, radon

class CT_dataset(torch.utils.data.Dataset):

    def __init__(self, directory, network = 'glimpse'):

        self.directory = directory

        self.name_list = sorted(os.listdir(self.directory))
        self.network = network


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        file = np.load(os.path.join(self.directory,file_name))
        image = file['image']
        image = torch.tensor(image, dtype = torch.float32)


        if self.network == 'unet':

            fbp = file['fbp']
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[None,...], fbp
        
        elif self.network == 'iradon':
            sinogram = file['sinogram']
            sinogram = torch.tensor(sinogram, dtype = torch.float32)

            return image[None,...], sinogram

        
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
        # print(image.shape, fbp.shape, sinogram.shape)

        ################ Remove it later
        # image = F.interpolate(image[None,...], size = image.shape[1]* (256//128),
        #                 mode = 'bilinear',
        #                 antialias= True,
        #                 align_corners= True)[0]

        
        # fbp = F.interpolate(fbp[None,...], size = fbp.shape[1]* (256//128),
        #                 mode = 'bilinear',
        #                 antialias= True,
        #                 align_corners= True)[0]

        
        # sinogram = F.interpolate(sinogram[None,...], size = (sinogram.shape[1],363),
        #                 mode = 'bilinear',
        #                 antialias= True,
        #                 align_corners= True)[0]
        
        # print(image.shape, fbp.shape, sinogram.shape)
            
        return image, sinogram, fbp


class CT_images(torch.utils.data.Dataset):

    def __init__(self, directory, image_size, theta_actual, theta_init,
                  noise_snr = 30, unet = False, subset = 'train'):

        self.directory = directory

        self.name_list = sorted(os.listdir(self.directory))
        self.image_size = image_size
        self.unet = unet
        self.theta_actual = theta_actual
        self.theta_init = theta_init
        self.noise_snr = noise_snr
        self.subset = subset


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        np.random.seed(idx + len(self.name_list))

        file_name = self.name_list[idx]
        if self.subset == 'ood':
            image = imageio.imread(os.path.join(self.directory,file_name))
            image = (image/255.0)

        elif self.subset == 'div2k':
            image = imageio.imread(os.path.join(self.directory,file_name))
            image = (image/255.0)
            # print(image.shape)
            image = np.mean(image, axis = 3)

        else:
            image = np.load(os.path.join(self.directory,file_name))
        
    

        image = torch.tensor(image, dtype = torch.float32)[None,None]
        image = F.interpolate(image, size = self.image_size,
                                mode = 'bilinear',
                                antialias= True,
                                align_corners= True)[0,0].cpu().detach().numpy()
        
        sinogram = radon(image, theta=self.theta_actual, circle= False)
        noise_sigma = 10**(-self.noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
        noise = np.random.normal(loc = 0,
                                 scale = noise_sigma,
                                 size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
        sinogram += noise

        image = torch.tensor(image, dtype = torch.float32)

        # print(image.shape, sinogram.shape)


        if self.unet:

            fbp = iradon(sinogram, theta= self.theta_init, circle= False)
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[None,...], fbp
        
        else:
            sinogram = torch.tensor(sinogram, dtype = torch.float32)
            image = image.reshape(-1, 1)
            return image, sinogram
        
