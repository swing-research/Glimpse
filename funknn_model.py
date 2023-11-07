import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import config_funknn as config
from skimage.transform.radon_transform import _get_fourier_filter
from utils import *



def reflect_coords(ix, min_val, max_val):

    pos_delta = ix[ix>max_val] - max_val

    neg_delta = min_val - ix[ix < min_val]

    ix[ix>max_val] = ix[ix>max_val] - 2*pos_delta
    ix[ix<min_val] = ix[ix<min_val] + 2*neg_delta

    return ix


def circular_coords(ix, min_val, max_val):

    ix[ix>max_val] = ix[ix>max_val] - (max_val - min_val)
    ix[ix<min_val] = ix[ix<min_val] + (max_val - min_val)

    return ix


class Deep_local(nn.Module):

    def __init__(self):
        super(Deep_local, self).__init__()

        fcs = []
        # prev_unit = config.w_size * config.w_size * (config.n_angles + 2*config.scale)
        # prev_unit = config.w_size * config.w_size * (config.n_angles + 2*(config.scale-1))
        if config.multiscale:
            prev_unit = config.w_size * config.w_size * (config.n_angles + 2*(config.scale-1))
        else:
            prev_unit = config.w_size * config.w_size * config.n_angles
        
        # hidden_units = [9,9,8,8,7,7,6,6,6,0] # medium
        # hidden_units = [10,10,10,9,9,9,8,8,0] # 512
        # hidden_units = [9,9,9,9,8,8,0] # med_shallow
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


        # Skip connection weight
        alpha = torch.zeros(1)
        # alpha = torch.ones(1)
        self.alpha = nn.Parameter(alpha.clone().detach(), requires_grad=False)

        n = int(np.ceil((config.image_size) * np.sqrt(2)))
        # projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n))))
        # projection_size_padded = config.image_size + 10
        projection_size_padded = 512
        # projection_size_padded = 1024
        fourier_filter = _get_fourier_filter(projection_size_padded, config.filter_init)
        fourier_filter = torch.tensor(fourier_filter, dtype = torch.float32)
        self.fourier_filter = nn.Parameter(fourier_filter.clone().detach(), requires_grad=config.learnable_filter) 

        s = (n-1)/2 * torch.ones(1)
        self.s = nn.Parameter(s.clone().detach(), requires_grad=True)

        # z = (torch.arange(config.n_angles) - (config.n_angles-1)/2)/((config.n_angles-1)/2)
        # self.z = nn.Parameter(z.clone().detach(), requires_grad=True)

        # theta_rad = torch.deg2rad(torch.tensor(
        #     config.theta[None,...,None, None], dtype = torch.float32))
        # self.theta_rad = nn.Parameter(theta_rad.clone().detach(), requires_grad=False)

        z = (torch.arange(config.n_angles) - (config.n_angles-1)/2)/((config.n_angles-1)/2)
        self.z = nn.Parameter(z.clone().detach(), requires_grad= config.lsg)

        theta_rad = torch.deg2rad(torch.tensor(
            config.theta[None,...,None, None], dtype = torch.float32))
        self.theta_rad = nn.Parameter(theta_rad.clone().detach(), requires_grad= config.lsg)


        if config.activation == 'sin':
            w0 = 30.0
            w = []
            for i in range(len(self.MLP)):
                w_init = torch.ones(1) * w0
                w.append(nn.Parameter(w_init.clone().detach(), requires_grad=True))
                w_shape = self.MLP[i].weight.data.shape
                b_shape = self.MLP[i].bias.data.shape
                w_std = (1 / w_shape[1]) if i==0 else (np.sqrt(6.0 / w_shape[1]) / w0)
                # w_std = (1 / w_shape[1])
                self.MLP[i].weight.data = (2 * torch.rand(w_shape) - 1) * w_std
                self.MLP[i].bias.data = (2 * torch.rand(b_shape) - 1) * w_std
            self.w = nn.ParameterList(w)



        
    def grab(self, coords, sinogram):
        # x: [b, n, 2]
        b = coords.shape[0]
        n = sinogram.shape[1]
        h = np.int(np.floor(n/np.sqrt(2)))

        coords = reflect_coords((coords + 0.5) * (h-1) , -0.5, h-1 + 0.5)
        coords = coords/(h-1) - 0.5

        col = sinogram.permute(0,2,1).unsqueeze(1)
        coords = coords.unsqueeze(1) * (h-1)
        xpr = coords[:,:,:,0]
        ypr = coords[:,:,:,1]
        
        # theta_rad = torch.deg2rad(torch.tensor(
        #     config.theta[None,...,None, None],
        #     dtype = torch.float32)).to(coords.device)
        theta_rad = self.theta_rad
        
        ypr = ypr/self.s
        xpr = xpr/self.s
        xpr = xpr.unsqueeze(1).repeat(1,config.n_angles,1,1)
        ypr = ypr.unsqueeze(1).repeat(1,config.n_angles,1,1)

        t = ypr * torch.cos(theta_rad) - xpr * torch.sin(theta_rad)
        t = t[...,None]
        # z = (torch.arange(config.n_angles).to(coords.device) - (config.n_angles-1)/2)/((config.n_angles-1)/2)
        z = self.z

        z = z[...,None,None,None]
        z = z[None,...].repeat(t.shape[0],1,t.shape[2], t.shape[3],1)
        t = torch.concat((t, z), dim = -1)
        t = t.reshape(b, config.n_angles * t.shape[2], t.shape[3], 2)
        cbp = F.grid_sample(col, t, align_corners= True, mode = 'bilinear')
        cbp = cbp.reshape(b, config.n_angles, t.shape[2])
        return cbp

        

    def sinogram_sampler(self, sinogram, coordinate , output_size):
        '''Cropper using Spatial Transformer'''
        # Coordinate shape: b X b_pixels X 2
        # image shape: b X c X h X w
        d_coordinate = coordinate * 2
        b , n , _ = sinogram.shape
        h = np.int(np.floor(n/np.sqrt(2)))

        b_pixels = coordinate.shape[1]
        crop_size = 2 * (output_size-1)/(h-1)
        x_m_x = crop_size/2
        x_p_x = d_coordinate[:,:,1]
        y_m_y = crop_size/2
        y_p_y = d_coordinate[:,:,0]
        theta = torch.zeros(b, b_pixels, 2,3).to(sinogram.device)
        theta[:,:,0,0] = x_m_x * self.ws1
        theta[:,:,0,2] = x_p_x
        theta[:,:,1,1] = y_m_y * self.ws2
        theta[:,:,1,2] = y_p_y

        theta = theta.reshape(b*b_pixels , 2 , 3)

        f = F.affine_grid(theta, size=(b * b_pixels, config.n_angles, output_size, output_size), align_corners=True)
        f = f.reshape(b, b_pixels , output_size, output_size,2)
        f = f.reshape(b, b_pixels * output_size, output_size,2).permute(0,3,1,2)
        f = f.reshape(b, 2, b_pixels * output_size * output_size).permute(0,2,1).flip(dims=[2])
        sinogram_samples = self.grab(f/2, sinogram)
        sinogram_samples = sinogram_samples.reshape(b, -1, b_pixels * output_size, output_size)

        sinogram_samples = sinogram_samples.permute(0,2,3,1)
        sinogram_samples = sinogram_samples.reshape(b, b_pixels , output_size, output_size,config.n_angles)
        sinogram_samples = sinogram_samples.reshape(b* b_pixels , output_size, output_size,config.n_angles)
        sinogram_samples = sinogram_samples.permute(0,3,1,2)

        return sinogram_samples
    
    
    def forward(self, coordinate, sinogram):

        # FBP cropper:
        # x_fbp = self.cropper(fbp , coordinate , output_size = config.w_size_fbp)
        # # mid_pix = x_fbp[:,:,4,4] # Centeric pixel
        # x_fbp = torch.flatten(x_fbp, 1)
        
        # Sinogram grabber
        b , n, _ = sinogram.shape
        projection_size_padded = 512
        # projection_size_padded = 1024
        pad_width = (0,0,0, projection_size_padded - n)
        padded_sinogram = F.pad(sinogram, pad_width)
        projection = torch.fft.fft(padded_sinogram, dim=1) * self.fourier_filter
        filtered_sinogram = torch.fft.ifft(projection, dim=1)[:,:n].real

        b , b_pixels , _ = coordinate.shape

        x_sin = self.sinogram_sampler(filtered_sinogram , coordinate , output_size = config.w_size)
        mid_pix = torch.mean(x_sin[:,:, config.w_size//2, config.w_size//2], dim = 1, keepdim=True) * np.pi/2 # FBP recon
        x = torch.flatten(x_sin, 1)

        if config.multiscale:
            # 1) low-scale features
            x_sin_fft = torch.fft.rfft(x_sin, dim = 1)

            scales = [2,4,8]
            for s in scales:

                x_sin_low = torch.fft.irfft(x_sin_fft[:,:s], dim = 1)
                x_sin_low = torch.flatten(x_sin_low, 1)
                x = torch.concat([x, x_sin_low], dim = 1)
            # print(x_sin_low.shape, x_sin.shape)

            # x_sin = torch.flatten(x_sin, 1)
            # x_sin_low = torch.flatten(x_sin_low, 1)
            # x = torch.concat([x_sin, x_sin_low], dim = 1)


            # 2) Fourior features
            # x_sin_fft = torch.fft.rfft(x_sin, dim = 1, norm = 'forward')
            # # print(x_sin_fft.shape, x_sin_fft[0,:,0,0])
            # x_sin_fft_real = x_sin_fft.real
            # x_sin_fft_imag = x_sin_fft.imag

            # x_sin = torch.flatten(x_sin, 1)
            # x_sin_fft_real = torch.flatten(x_sin_fft_real[:,:config.scale], 1)
            # x_sin_fft_imag = torch.flatten(x_sin_fft_imag[:,:config.scale], 1)

            # x = torch.concat([x_sin, x_sin_fft_real, x_sin_fft_imag], dim = 1)

        for i in range(len(self.MLP)-1):
            if config.activation == 'relu':
                x = F.relu(self.MLP[i](x))
            elif config.activation == 'sin':
                x = torch.sin(self.w[i] * self.MLP[i](x))

        x = self.MLP[-1](x)
        x = x + self.alpha * mid_pix # external skip connection to the centric pixel
        x = x.reshape(b, b_pixels, -1)

        return x

