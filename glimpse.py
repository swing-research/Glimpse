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

    def __init__(self, n1, n2, n3, patch_shape, learned_geo, theta_init, lsg,
                 learnable_filter, filter_init):
        super(glimpse, self).__init__()

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.lsg = lsg
        self.learnable_filter = learnable_filter
        self.filter_init = filter_init
        self.n_angles = len(theta_init)
        self.patch_shape = patch_shape
        self.learned_geo = learned_geo

        self.N1 = 8
        self.N2 = 8
        self.N3 = 3

        # self.N1 = 1
        # self.N2 = 1
        # self.N3 = 1


        # fcs = []
        # prev_unit = self.N1 * self.N2 * self.N3 * 41
        # hidden_units = [7,7,7,7,0]
        # hidden_units = np.power(2, hidden_units)

        # for i in range(len(hidden_units)):
        #     fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = False))
        #     prev_unit = hidden_units[i]
        #     # if i < len(hidden_units)-1:
        #     #     fcs.append(nn.ReLU())
        
        # # self.MLP = nn.Sequential(*fcs)
        # self.MLP = nn.ModuleList(fcs)
        
        fcs1 = []
        prev_unit = self.N1 * self.N2 * self.N3 * 11
        hidden_units = [10,9,9,8,7]
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs1.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]
            if i < len(hidden_units)-1:
                fcs1.append(nn.ReLU())

        fcs2 = []
        prev_unit = self.N1 * self.N2 * self.N3 * 10
        hidden_units = [10,9,9,8,7]
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs2.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]
            if i < len(hidden_units)-1:
                fcs2.append(nn.ReLU())

        fcs3 = []
        prev_unit = self.N1 * self.N2 * self.N3 * 10
        hidden_units = [10,9,9,8,7]
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs3.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]
            if i < len(hidden_units)-1:
                fcs3.append(nn.ReLU())

        fcs4 = []
        prev_unit = self.N1 * self.N2 * self.N3 * 10
        hidden_units = [10,9,9,8,7]
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs4.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]
            if i < len(hidden_units)-1:
                fcs4.append(nn.ReLU())

        
        fcs_agg = []
        prev_unit = 4 * 128
        hidden_units = [9,9,8,7,0]
        hidden_units = np.power(2, hidden_units)

        for i in range(len(hidden_units)):
            fcs_agg.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]
            if i < len(hidden_units)-1:
                fcs_agg.append(nn.ReLU())
        

        self.MLP1 = nn.Sequential(*fcs1)
        self.MLP2 = nn.Sequential(*fcs2)
        self.MLP3 = nn.Sequential(*fcs3)
        self.MLP4 = nn.Sequential(*fcs4)
        self.MLP_agg = nn.Sequential(*fcs_agg)

        fourier_filter = _get_fourier_filter(2*n1, self.filter_init)
        fourier_filter = fourier_filter[:,0]
        fourier_filter = torch.tensor(fourier_filter, dtype = torch.float32)
        self.fourier_filter = nn.Parameter(fourier_filter.clone().detach(), requires_grad=self.learnable_filter) 

        z = (torch.arange(self.n_angles) - (self.n_angles-1)/2)/((self.n_angles-1)/2)
        self.z = nn.Parameter(z.clone().detach(), requires_grad= self.lsg)

        theta_rad = torch.tensor(
            theta_init[None,...,None, None], dtype = torch.float32)
        self.theta_rad = nn.Parameter(theta_rad.clone().detach(), requires_grad= self.lsg)

        if self.patch_shape == 'round':

            r = self.N/self.image_size
            thetas = torch.arange(self.M)*(2*np.pi/self.M)
            x = r*torch.cos(thetas)/(2*self.N)
            y = r*torch.sin(thetas)/(2*self.N)
            x = x[...,None]
            y = y[...,None]
            xy = torch.concat([x,y], dim = 1)[None,...]
            xy = xy.expand(self.N,-1,-1)
            idx = (torch.arange(0,self.N))[...,None,None]
            xy = idx * xy

        elif self.patch_shape == 'square':

            x = torch.arange(-(self.N//2), self.N//2+1)/(self.image_size)
            y = torch.arange(-(self.M//2), self.M//2+1)/(self.image_size)
            x , y = torch.meshgrid(x,y, indexing='ij')
            x = x[...,None]
            y = y[...,None]
            xy = torch.concat([x,y], dim = 2)[None,...]

        elif self.patch_shape == 'random':

            xy = 2 * self.N1*(torch.rand(self.N1, self.N2, self.N3 ,3) - 0.5)/(self.n1)
            xy[0,0,0] = 0
            # xy = xy * 0

        self.xy = nn.Parameter(xy.clone().detach(), requires_grad=self.learned_geo)

        # Adaptive receptive field
        recep_scale = 1
        patch_scale = recep_scale*torch.ones(1)
        self.patch_scale = nn.Parameter(patch_scale.clone().detach(), requires_grad=True)

        alpha = torch.ones(1)
        self.alpha = nn.Parameter(alpha.clone().detach(), requires_grad=True)


    
    def extract_sin(self, coords, sinogram):

        b = coords.shape[0]
        coords = reflect_coords(coords, -1, 1)

        coords = coords.unsqueeze(1)
        xpr = coords[:,:,:,0]
        ypr = coords[:,:,:,1]
        zpr = coords[:,:,:,2]
        
        theta_rad = self.theta_rad

        xpr = xpr.unsqueeze(1).repeat(1,self.n_angles,1,1)
        ypr = ypr.unsqueeze(1).repeat(1,self.n_angles,1,1)
        zpr = zpr.unsqueeze(1).repeat(1,self.n_angles,1,1)

        t = ypr * torch.cos(theta_rad) + zpr * torch.sin(theta_rad)
        t = t[...,None]
        xpr = xpr[...,None]
        z = self.z
        z = z[...,None,None,None]
        z = z[None,...].repeat(t.shape[0],1,t.shape[2], t.shape[3],1)
        t = torch.concat((t ,xpr ,z), dim = -1)
        sinogram = sinogram.unsqueeze(1)
        cbp = F.grid_sample(sinogram, t, align_corners= True, mode = 'bilinear')
        cbp = cbp.reshape(b, self.n_angles, -1)
        return cbp
    


    def sinogram_sampler(self, sinogram, coordinate):
        '''Sinosuidal Extraction'''
        # Coordinate shape: b X b_pixels X 3
        # Sinogram shape: b X n_proj X n1 X n2
        b , _ , n1 , _ = sinogram.shape
        b_pixels = coordinate.shape[1]
        xy = self.patch_scale * self.xy / (n1/self.n1)
        xy = xy[None, None]
        N1 = self.N1
        N2 = self.N2
        N3 = self.N3
        coordinate = coordinate.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        f = coordinate + xy # b x b_pixels x N1 x N2 x N3 x 3

        f = f.reshape(b, b_pixels * N1 * N2 * N3, 3)
        sinogram_samples = self.extract_sin(f, sinogram)
        sinogram_samples = sinogram_samples.permute(0,2,1)
        sinogram_samples = sinogram_samples.reshape(b, b_pixels , N1, N2 , N3, -1)
        sinogram_samples = sinogram_samples.reshape(b* b_pixels , N1, N2, N3, -1)

        return sinogram_samples
        

    def forward(self, coordinate, proj):
        b , b_pixels, _ = coordinate.shape

        # if not proj.shape[2] == 2*self.fourier_filter.shape[0]:

        #     fourier_filter = F.interpolate(self.fourier_filter[None,None,...], size = 2*proj.shape[2])[0,0]

        # else:
        #     fourier_filter = self.fourier_filter

        fourier_filter = self.fourier_filter
        filtered_proj = custom_ramp_fft(proj,fourier_filter)

        x_sin = self.sinogram_sampler(filtered_proj , coordinate) * 700

        # x = x_sin[:,0,0,0]
        # return torch.mean(x, dim = -1)
    
        # for i in range(len(self.MLP)-1):
        #     x = F.relu(self.MLP[i](x))
        # p = self.MLP[-1](x)


        p1 = torch.flatten(x_sin[:,:,:,:,:11],1)
        p1 = self.MLP1(p1)
        p2 = torch.flatten(x_sin[:,:,:,:,11:21],1)
        p2 = self.MLP2(p2)
        p3 = torch.flatten(x_sin[:,:,:,:,21:31],1)
        p3 = self.MLP3(p3)
        p4 = torch.flatten(x_sin[:,:,:,:,31:],1)
        p4 = self.MLP4(p4)

        p = torch.concat((p1,p2,p3,p4), dim = 1)
        p = self.MLP_agg(p)
        # skip = torch.mean(x_sin[:,0,0,0], dim = -1, keepdim=True)
        # p = -p + skip * 660
        p = p.reshape(b, b_pixels) 

        return p
