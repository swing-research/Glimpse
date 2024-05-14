import torch
from utils import *
import numpy as np
import os
import imageio
import torch.nn.functional as F
from skimage.transform import iradon, radon
from torch.utils.data import Dataset
import mrcfile
from simulator import Simulator
import config

class CT_dataset(Dataset):

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
        
    
    

class CT_odl(Dataset):

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



class CT_images(Dataset):

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


        if self.unet:

            fbp = iradon(sinogram, theta= self.theta_init, circle= False)
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[None,...], fbp
        
        else:
            sinogram = torch.tensor(sinogram, dtype = torch.float32)
            image = image.reshape(-1, 1)
            return image, sinogram
        



class simulatorVolumes(Dataset):
    """
    Loads the volumes from the simulator dataset
    """
    def __init__(self, root_dir: str ,
                 models: list,
                 normalize_type: str = 'vol',
                 negate_volume: bool = True,
                 n1: int = 1024):
        """
        Args:
            root_dir (string): Directory with all the folders.
            models (list): list of models to load 
            normalize_type (str): type of normalization to use (vol,  none)
            
        TODO: Add more projection types
        """

        self.root_dir = root_dir
        self.models = models
        self.normalize_type = normalize_type
        self.negate_volume = negate_volume
        self.n1 = n1



        self.files = []
        for model in self.models:
            self.files += [os.path.join(self.root_dir, model)]
        #print(self.files)

        self.vol_file_names, self.proj_file_names = self.get_file_names()


    def get_file_names(self):
        """
        Get all the file names of the volumes and projections
        used 
        TODO: might be suboptimal for very large datasets. But I am not planning to 
        go beyond 100 models
        """
        vol_file_names = []
        proj_file_names = []

        for file in self.files:
            #print(file)
            path = file
            # Find the volume file name it should start with 0_model

            for file_name in os.listdir(path):
                if file_name.endswith('.mrc'):
                    vol_file_names.append(os.path.join(path, file_name))
                

        return vol_file_names, proj_file_names


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vol_path = self.vol_file_names[idx]
        vol = mrcfile.read(vol_path)
        #print(vol.shape)
        vol = np.moveaxis(vol, 0, 2)
        if self.negate_volume:
            vol = vol_normalize(-vol,vol.min(), vol.max())

        # Normalize the data
        if self.normalize_type == 'vol':
            max_value = np.max(abs(vol))
            vol = vol/max_value
        if self.normalize_type == 'standard':
            vol = (vol - np.mean(vol))/np.std(vol)

            vol = vol/np.max(abs(vol))
        elif self.normalize_type == 'none':
            pass

        vol = torch.FloatTensor(vol).permute(2,0,1)

        vol = F.interpolate(vol[None,...], size = self.n1,
                        mode = 'bilinear',
                        antialias= True,
                        align_corners= True)[0].permute(1,2,0)


        return vol
    

def vol_normalize(vol,min_Val,max_Val):
    vol = (vol- np.min(vol))/(np.max(vol)-np.min(vol))
    vol = vol*(max_Val-min_Val)+min_Val
    return vol



class pixel_loader(Dataset):

    def __init__(self, vol, coords, n_samples = 10000):

        self.vol = vol
        self.coords = coords
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        pixels = np.random.randint(low = 0, high = config.n1*config.n2*config.n3)
        batch_coords = self.coords[pixels,:]
        batch_image = self.vol[pixels,:]

        batch_coords = torch.tensor(batch_coords, dtype = torch.float32)
        batch_image = torch.tensor(batch_image, dtype = torch.float32)

        return batch_coords, batch_image