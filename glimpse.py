import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from skimage.transform.radon_transform import _get_fourier_filter
from utils import *



def reflect_coords(ix, min_val, max_val):

    pos_delta = ix[ix>max_val] - max_val

    neg_delta = min_val - ix[ix < min_val]

    ix[ix>max_val] = ix[ix>max_val] - 2*pos_delta
    ix[ix<min_val] = ix[ix<min_val] + 2*neg_delta

    return ix


class glimpse(nn.Module):

    def __init__(self, image_size, w_size, theta_init, lsg,
                 learnable_filter, filter_init):
        super(glimpse, self).__init__()

        self.image_size = image_size
        self.w_size = w_size
        self.lsg = lsg
        self.learnable_filter = learnable_filter
        self.filter_init = filter_init
        self.n_angles = len(theta_init)

        fcs = []
        
        prev_unit = self.w_size * self.w_size * self.n_angles
        hidden_units = [8,8,8,8,7,7,7,6,6,0] # base
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]

        self.MLP = nn.ModuleList(fcs)

        # Adaptive receptive field
        ws1 = torch.ones(1)
        self.ws1 = nn.Parameter(ws1.clone().detach(), requires_grad=True)
        ws2 = torch.ones(1)
        self.ws2 = nn.Parameter(ws2.clone().detach(), requires_grad=True)


        n = int(np.ceil((self.image_size) * np.sqrt(2)))
        projection_size_padded = 512
        # projection_size_padded = 1024
        fourier_filter = _get_fourier_filter(projection_size_padded, self.filter_init)
        fourier_filter = torch.tensor(fourier_filter, dtype = torch.float32)
        self.fourier_filter = nn.Parameter(fourier_filter.clone().detach(), requires_grad=self.learnable_filter) 

        s = (n-1)/2 * torch.ones(1)
        self.s = nn.Parameter(s.clone().detach(), requires_grad=True)

        z = (torch.arange(self.n_angles) - (self.n_angles-1)/2)/((self.n_angles-1)/2)
        self.z = nn.Parameter(z.clone().detach(), requires_grad= self.lsg)

        theta_rad = torch.deg2rad(torch.tensor(
            theta_init[None,...,None, None], dtype = torch.float32))
        self.theta_rad = nn.Parameter(theta_rad.clone().detach(), requires_grad= self.lsg)

        
    def extract_sin(self, coords, sinogram):

        b = coords.shape[0]
        n = sinogram.shape[1]
        h = np.int32(np.floor(n/np.sqrt(2)))

        coords = reflect_coords((coords + 0.5) * (h-1) , -0.5, h-1 + 0.5)
        coords = coords/(h-1) - 0.5

        col = sinogram.permute(0,2,1).unsqueeze(1)
        coords = coords.unsqueeze(1) * (h-1)
        xpr = coords[:,:,:,0]
        ypr = coords[:,:,:,1]
        
        theta_rad = self.theta_rad
        
        ypr = ypr/self.s
        xpr = xpr/self.s
        xpr = xpr.unsqueeze(1).repeat(1,self.n_angles,1,1)
        ypr = ypr.unsqueeze(1).repeat(1,self.n_angles,1,1)

        t = ypr * torch.cos(theta_rad) - xpr * torch.sin(theta_rad)
        t = t[...,None]
        z = self.z

        z = z[...,None,None,None]
        z = z[None,...].repeat(t.shape[0],1,t.shape[2], t.shape[3],1)
        t = torch.concat((t, z), dim = -1)
        t = t.reshape(b, self.n_angles * t.shape[2], t.shape[3], 2)
        cbp = F.grid_sample(col, t, align_corners= True, mode = 'bilinear')
        cbp = cbp.reshape(b, self.n_angles, t.shape[2])
        return cbp

        

    def sinogram_sampler(self, sinogram, coordinate , output_size):
        '''Cropper using Spatial Transformer'''

        d_coordinate = coordinate * 2
        b , n , _ = sinogram.shape
        h = np.int32(np.floor(n/np.sqrt(2)))

        b_pixels = coordinate.shape[1]
        crop_size = 2 * (output_size-1)/(h-1)
        x_m_x = crop_size/2
        x_p_x = d_coordinate[:,:,1]
        y_m_y = crop_size/2
        y_p_y = d_coordinate[:,:,0]
        affine_mat = torch.zeros(b, b_pixels, 2,3).to(sinogram.device)
        affine_mat[:,:,0,0] = x_m_x * self.ws1
        affine_mat[:,:,0,2] = x_p_x
        affine_mat[:,:,1,1] = y_m_y * self.ws2
        affine_mat[:,:,1,2] = y_p_y

        affine_mat = affine_mat.reshape(b*b_pixels , 2 , 3)

        f = F.affine_grid(affine_mat, size=(b * b_pixels, self.n_angles, output_size, output_size), align_corners=True)
        f = f.reshape(b, b_pixels , output_size, output_size,2)
        f = f.reshape(b, b_pixels * output_size, output_size,2).permute(0,3,1,2)
        f = f.reshape(b, 2, b_pixels * output_size * output_size).permute(0,2,1).flip(dims=[2])
        sinogram_samples = self.extract_sin(f/2, sinogram)
        sinogram_samples = sinogram_samples.reshape(b, -1, b_pixels * output_size, output_size)

        sinogram_samples = sinogram_samples.permute(0,2,3,1)
        sinogram_samples = sinogram_samples.reshape(b, b_pixels , output_size, output_size,self.n_angles)
        sinogram_samples = sinogram_samples.reshape(b* b_pixels , output_size, output_size,self.n_angles)
        sinogram_samples = sinogram_samples.permute(0,3,1,2)

        return sinogram_samples
    
    
    def forward(self, coordinate, sinogram):
        
        # Sinogram grabber
        b , n, _ = sinogram.shape
        projection_size_padded = 512
        # projection_size_padded = 1024
        pad_width = (0,0,0, projection_size_padded - n)
        padded_sinogram = F.pad(sinogram, pad_width)
        projection = torch.fft.fft(padded_sinogram, dim=1) * self.fourier_filter
        filtered_sinogram = torch.fft.ifft(projection, dim=1)[:,:n].real

        b , b_pixels , _ = coordinate.shape

        x_sin = self.sinogram_sampler(filtered_sinogram , coordinate , output_size = self.w_size)

        x = torch.flatten(x_sin, 1)
        for i in range(len(self.MLP)-1):
            x = F.relu(self.MLP[i](x))

        x = self.MLP[-1](x)
        x = x.reshape(b, b_pixels, -1)
        x = x * np.pi/2

        return x
