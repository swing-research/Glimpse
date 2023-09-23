import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import config_funknn as config
from skimage.transform.radon_transform import _get_fourier_filter
import bayes as bnn
from utils import *


def squeeze(x , f):
    x = x.permute(0,2,3,1)
    b, N1, N2, nch = x.shape
    x = torch.reshape(
        torch.permute(
            torch.reshape(x, shape=[b, N1//f, f, N2//f, f, nch]),
            [0, 1, 3, 2, 4, 5]),
        [b, N1//f, N2//f, nch*f*f])
    x = x.permute(0,3,1,2)
    return x


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

 

def cubic_coords(ix,iy, indices =  [-1,0,1,2]):

    with torch.no_grad():
        ix_base = torch.floor(ix)
        iy_base = torch.floor(iy)
        points = torch.zeros(ix.shape + (len(indices)**2,2)).to(ix.device)
        for i in range(len(indices)):
            for j in range(len(indices)):
                points[...,len(indices) *i + j,0] = indices[i] + ix_base
                points[...,len(indices) *i + j,1] = indices[j] + iy_base
    
    return points


def cubic_kernel(s1, order = 4, second_order = False):
    s = s1 + 1e-6
    out = torch.zeros_like(s)

    if second_order == False:
        if order == 4:
            # p = torch.abs(s[torch.abs(s)== 2])
            # out[torch.abs(s)== 2] = (-0.5 * p**3 + 2.5 * p**2 -4*p + 2)/2

            p = torch.abs(s[torch.abs(s)< 2])
            out[torch.abs(s)< 2] = -0.5 * p**3 + 2.5 * p**2 -4*p + 2

            p = torch.abs(s[torch.abs(s)== 1])
            out[torch.abs(s)== 1] = ((1.5 * p**3 - 2.5 * p**2 + 1) + 3*(-0.5 * p**3 + 2.5 * p**2 -4*p + 2))/4

            p = torch.abs(s[torch.abs(s)< 1])
            p_wo_abs = s[torch.abs(s)< 1]
            out[torch.abs(s)< 1] = 1.5 * p**3 - 2.5 * p_wo_abs**2 + 1

        elif order == 6:
            p = torch.abs(s[torch.abs(s)< 3])
            out[torch.abs(s)< 3] = (1 * p**3)/12 - (2 * p**2)/3 + 21*p/12 - 1.5
            p = torch.abs(s[torch.abs(s)< 2])
            out[torch.abs(s)< 2] = -(7 * p**3)/12 + 3 * p**2 -59*p/12 + 15/6
            p = torch.abs(s[torch.abs(s)< 1])
            out[torch.abs(s)< 1] = (4 * p**3)/3 - (7 * p**2)/3 + 1

    else:
        if order == 4:
            a = 1
            p = torch.abs(s[torch.abs(s)< 2])
            out[torch.abs(s)< 2] = a * p**4 + (-71/13 * a) * p**3 + (8.76*a) * p**2 + (-1.53*a)*p - 1.76*a
            p = torch.abs(s[torch.abs(s)< 1])
            out[torch.abs(s)< 1] = (34.94 * a -6) * p**4 + (-79.41*a - 10) * p**3 + (44.47*a - 9) * p**2 + 1

        if order == 2:
            a = 5.5
            p = torch.abs(s[torch.abs(s)< 1])
            out[torch.abs(s)< 1] = (a) * p**5 + (3 - 3*a) * p**4 + (3*a-4) * p**3 + (-a) * p**2 + 1


    return out



class FunkNN(nn.Module):
    '''FunkNN module'''

    def __init__(self, c):
        super(FunkNN, self).__init__()
        
        self.c = c

        fcs = []
        prev_unit = config.w_size * config.w_size * self.c #+ 100
        # hidden_units = [10,10,10,10,9,9,9,9,8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,0]
        hidden_units = [9,9,8,8,7,7,6,6,6,0]
        hidden_units = np.power(2, hidden_units)

        if config.Bayesian:
            for i in range(len(hidden_units)):
                # if i > 7 and i < 14:
                if i > -1:
                    fcs.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                            in_features=prev_unit,
                                            out_features=hidden_units[i],
                                            sigma_init= 0.1))
                else:
                    fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = True))

                prev_unit = hidden_units[i]

        else:
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
        alpha = torch.ones(1)
        self.alpha = nn.Parameter(alpha.clone().detach(), requires_grad=True)

        n = int(np.ceil((config.image_size+1) * np.sqrt(2)))
        # projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n))))
        projection_size_padded = 512
        fourier_filter = _get_fourier_filter(projection_size_padded, config.filter_init)
        fourier_filter = torch.tensor(fourier_filter, dtype = torch.float32)
        self.fourier_filter = nn.Parameter(fourier_filter.clone().detach(), requires_grad=config.learnable_filter) 

        th = torch.zeros(1)
        self.th = nn.Parameter(th.clone().detach(), requires_grad=True)

        s = (n-1)/2 * torch.ones(1)
        self.s = nn.Parameter(s.clone().detach(), requires_grad=True)


        
    def grab(self, coords, sinogram):
        # x: [b, n, 2]
        b = coords.shape[0]
        n = sinogram.shape[1]
        n_base = int(np.ceil((config.image_size+1) * np.sqrt(2)))
        h = np.int(np.floor(n/np.sqrt(2)))

        coords = reflect_coords((coords + 0.5) * (h-1) , -0.5, h-1 + 0.5)
        coords = coords/(h-1) - 0.5
        # coords = 2 * reflect_coords(coords , -0.5 + self.th, 0.5 - self.th)
        # coords = coords/np.sqrt(2)

        col = sinogram.permute(0,2,1).unsqueeze(1)
        coords = coords.unsqueeze(1) * (h-1)
        xpr = coords[:,:,:,0]
        ypr = coords[:,:,:,1]
        
        theta_rad = torch.deg2rad(torch.tensor(
            config.theta[None,...,None, None],
            dtype = torch.float32)).to(coords.device)
        
        ypr = ypr/(self.s * (n-1)/(n_base-1))
        xpr = xpr/(self.s * (n-1)/(n_base-1))
        xpr = xpr.unsqueeze(1).repeat(1,config.c,1,1)
        ypr = ypr.unsqueeze(1).repeat(1,config.c,1,1)

        t = ypr * torch.cos(theta_rad) - xpr * torch.sin(theta_rad)
        # print(t.min(), t.max())
        t = t[...,None]
        z = (torch.arange(config.c).to(coords.device) - (config.c-1)/2)/((config.c-1)/2)
        z = z[...,None,None,None]
        z = z[None,...].repeat(t.shape[0],1,t.shape[2], t.shape[3],1)
        t = torch.concat((t, z), dim = -1)
        t = t.reshape(b, config.c * t.shape[2], t.shape[3], 2)
        cbp = F.grid_sample(col, t, align_corners= True, mode = 'bilinear')
        cbp = cbp.reshape(b, config.c, t.shape[2])
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

        f = F.affine_grid(theta, size=(b * b_pixels, config.c, output_size, output_size), align_corners=True)
        f = f.reshape(b, b_pixels , output_size, output_size,2)
        f = f.reshape(b, b_pixels * output_size, output_size,2).permute(0,3,1,2)
        f = f.reshape(b, 2, b_pixels * output_size * output_size).permute(0,2,1).flip(dims=[2])
        sinogram_samples = self.grab(f/2, sinogram)
        sinogram_samples = sinogram_samples.reshape(b, -1, b_pixels * output_size, output_size)

        sinogram_samples = sinogram_samples.permute(0,2,3,1)
        sinogram_samples = sinogram_samples.reshape(b, b_pixels , output_size, output_size,config.c)
        sinogram_samples = sinogram_samples.reshape(b* b_pixels , output_size, output_size,config.c)
        sinogram_samples = sinogram_samples.permute(0,3,1,2)

        return sinogram_samples
    
    
    def forward(self, coordinate, sinogram, factor = 1):
        
        b , n, _ = sinogram.shape
        projection_size_padded = 512

        pad_width = (0,0,0, projection_size_padded - n)
        padded_sinogram = F.pad(sinogram, pad_width)
        projection = torch.fft.fft(padded_sinogram, dim=1) * self.fourier_filter
        filtered_sinogram = torch.fft.ifft(projection, dim=1)[:,:n].real

        b , b_pixels , _ = coordinate.shape

        x = self.sinogram_sampler(filtered_sinogram , coordinate , output_size = config.w_size)
        mid_pix = torch.mean(x[:,:, config.w_size//2, config.w_size//2], dim = 1, keepdim=True) * np.pi/2 # FBP recon

        x = torch.flatten(x, 1)
        # coordinate = coordinate * 64
        # coordinate = coordinate - torch.floor(coordinate)
        # # print(coordinate)
        # coordinate = coordinate.reshape(b*b_pixels, -1).repeat(1,50)
        # # print(coordinate.shape)
        # x = torch.cat((coordinate, x), axis = 1)

        for i in range(len(self.MLP)-1):
            x = F.relu(self.MLP[i](x))

        x = self.MLP[-1](x)
        # x = x + self.alpha * mid_pix # external skip connection to the centric pixel
        x = x.reshape(b, b_pixels, -1)
        # x + self.alpha *
        
        return x

