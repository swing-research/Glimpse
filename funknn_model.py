import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import config_funknn as config
from skimage.transform.radon_transform import _get_fourier_filter
import bayes as bnn

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
        prev_unit = config.w_size * config.w_size * self.c #+ 2
        hidden_units = [10,10,10,10,9,9,9,9,8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,0]
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

        n = int(np.ceil(config.image_size * np.sqrt(2)))
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n))))
        fourier_filter = _get_fourier_filter(projection_size_padded, config.filter_init)
        fourier_filter = torch.tensor(fourier_filter, dtype = torch.float32)
        self.fourier_filter = nn.Parameter(fourier_filter.clone().detach(), requires_grad=config.learnable_filter)

        if config.missing_cone == 'horizontal':
            theta = np.linspace(30.0, 150.0, config.c, endpoint=False)

        elif config.missing_cone == 'vertical':
            theta = np.linspace(-60.0, 60.0, config.c, endpoint=False)

        else:
            theta = np.linspace(0.0, 180.0, config.c, endpoint=False)
        

        s = (n-1)/2
        theta_rad = torch.deg2rad(torch.tensor(theta[...,None, None], dtype = torch.float32))
        xpr, ypr = np.mgrid[:config.image_size, :config.image_size] - ((config.image_size)/2)
        ypr = ypr/s
        xpr = xpr/s
        xpr = torch.tensor(xpr[None,...].repeat(config.c, axis = 0), dtype = torch.float32)
        ypr = torch.tensor(ypr[None,...].repeat(config.c, axis = 0), dtype = torch.float32)

        t = ypr * torch.cos(theta_rad) - xpr * torch.sin(theta_rad)
        t = t[None,...]
        t = t[...,None]
        z = (torch.arange(config.c) - (config.c-1)/2)/((config.c-1)/2)
        z = z[...,None,None,None]
        z = z[None,...].repeat(1,1,config.image_size, config.image_size,1)
        t = torch.concat((t, z), dim = -1)
        t = t.reshape(1, config.c * config.image_size, config.image_size, 2)
        self.t = nn.Parameter(t.clone().detach(), requires_grad=True)


    def grid_sample_customized(self, image, grid, mode = 'cubic_conv' , pad = 'reflect', align_corners = False):
        '''Differentiable grid_sample:
        equivalent performance with torch.nn.functional.grid_sample can be obtained by setting
        align_corners = True,
        pad: 'border': use border pixels,
        'reflect': create reflect pad manually.
        image is a tensor of shape (N, C, IH, IW)
        grid is a tensor of shape (N, H, W, 2)'''

        if mode == 'bilinear':
            N, C, IH, IW = image.shape
            _, H, W, _ = grid.shape

            ix = grid[..., 0]
            iy = grid[..., 1]


            if align_corners == True:
                ix = ((ix + 1) / 2) * (IW-1);
                iy = ((iy + 1) / 2) * (IH-1);
                
                boundary_x = (0, IW-1)
                boundary_y = (0, IH-1)
                
            
            elif align_corners == False:
                # ix = ((1+ix)*IW/2) - 1/2
                # iy = ((1+iy)*IH/2) - 1/2
                
                # boundary_x = (-1/2, IW-1/2)
                # boundary_y = (-1/2, IH-1/2)

                ix = ((1+ix)*IW/2) 
                iy = ((1+iy)*IH/2)
                
                boundary_x = (0, IW-1)
                boundary_y = (0, IH-1)
            

            with torch.no_grad():
                ix_nw = torch.floor(ix);
                iy_nw = torch.floor(iy);
                ix_ne = ix_nw + 1;
                iy_ne = iy_nw;
                ix_sw = ix_nw;
                iy_sw = iy_nw + 1;
                ix_se = ix_nw + 1;
                iy_se = iy_nw + 1;

            nw = (ix_se - ix)    * (iy_se - iy)
            ne = (ix    - ix_sw) * (iy_sw - iy)
            sw = (ix_ne - ix)    * (iy    - iy_ne)
            se = (ix    - ix_nw) * (iy    - iy_nw)


            if pad == 'reflect' or 'reflection':
                
                ix_nw = reflect_coords(ix_nw, boundary_x[0], boundary_x[1])
                iy_nw = reflect_coords(iy_nw, boundary_y[0], boundary_y[1])

                ix_ne = reflect_coords(ix_ne, boundary_x[0], boundary_x[1])
                iy_ne = reflect_coords(iy_ne, boundary_y[0], boundary_y[1])

                ix_sw = reflect_coords(ix_sw, boundary_x[0], boundary_x[1])
                iy_sw = reflect_coords(iy_sw, boundary_y[0], boundary_y[1])

                ix_se = reflect_coords(ix_se, boundary_x[0], boundary_x[1])
                iy_se = reflect_coords(iy_se, boundary_y[0], boundary_y[1])


            elif pad == 'border':

                with torch.no_grad():
                    torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
                    torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

                    torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
                    torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

                    torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
                    torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

                    torch.clamp(ix_se, 0, IW-1, out=ix_se)
                    torch.clamp(iy_se, 0, IH-1, out=iy_se)


            image = image.reshape(N, C, IH * IW)

            nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
            ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
            sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
            se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

            out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
                    ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
                    sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
                    se_val.view(N, C, H, W) * se.view(N, 1, H, W))


            return out_val


        elif mode == 'cubic_conv':

            N, C, IH, IW = image.shape
            _, H, W, _ = grid.shape

            ix = grid[..., 0]
            iy = grid[..., 1]


            if align_corners == True:
                ix = ((ix + 1) / 2) * (IW-1);
                iy = ((iy + 1) / 2) * (IH-1);
                
                boundary_x = (0, IW-1)
                boundary_y = (0, IH-1)
                
            
            elif align_corners == False:
                ix = ((1+ix)*IW/2) - 1/2
                iy = ((1+iy)*IH/2) - 1/2
                
                boundary_x = (-1/2, IW-1/2)
                boundary_y = (-1/2, IH-1/2)
            
            # indices = [-2,-1,0,1,2,3] # order 6
            indices = [-1,0,1,2] # order 4
            # indices = [0,1] # order 2
            points = cubic_coords(ix,iy, indices)
            n_neighrbours = len(indices)**2

            ix_relative = ix.unsqueeze(dim = -1)- points[...,0]
            iy_relative = iy.unsqueeze(dim = -1)- points[...,1]

            points[...,0] = reflect_coords(points[...,0], boundary_x[0], boundary_x[1])
            points[...,1] = reflect_coords(points[...,1], boundary_y[0], boundary_y[1])
            points = points.unsqueeze(dim = 1).expand(-1,C,-1,-1,-1,-1)

            image = image.reshape(N, C, IH * IW)
            points_values = torch.gather(image,2, (points[...,1] * IW + points[...,0]).long().view(N,C,H*W*n_neighrbours))
            points_values = points_values.reshape(N,C,H,W,n_neighrbours)

            ux = cubic_kernel(ix_relative, order = len(indices)).unsqueeze(dim = 1)
            uy = cubic_kernel(iy_relative, order = len(indices)).unsqueeze(dim = 1)

            recons = points_values * ux * uy
            out_val = torch.sum(recons, dim = 4)

            return out_val
        
    
    def cropper(self, image, coordinate , output_size):
        '''Cropper using Spatial Transformer'''
        # Coordinate shape: b X b_pixels X 2
        # image shape: b X c X h X w
        d_coordinate = coordinate * 2
        b , c , h , w = image.shape
        b_pixels = coordinate.shape[1]
        crop_size = 2 * (output_size-1)/(h-1)
        x_m_x = crop_size/2
        x_p_x = d_coordinate[:,:,1]
        y_m_y = crop_size/2
        y_p_y = d_coordinate[:,:,0]
        theta = torch.zeros(b, b_pixels, 2,3).to(image.device)
        theta[:,:,0,0] = x_m_x * self.ws1
        theta[:,:,0,2] = x_p_x
        theta[:,:,1,1] = y_m_y * self.ws2
        theta[:,:,1,2] = y_p_y

        theta = theta.reshape(b*b_pixels , 2 , 3)

        f = F.affine_grid(theta, size=(b * b_pixels, c, output_size, output_size), align_corners=False)
        f = f.reshape(b, b_pixels , output_size, output_size,2)
        f = f.reshape(b, b_pixels * output_size, output_size,2)

        # Non-differentiable grid_sampler
        image_cropped = F.grid_sample(image, f, mode = 'bicubic', align_corners=False, padding_mode='reflection')
        # Differentiable grid_sampler, well-suited for solving PDEs
        # image_cropped = self.grid_sample_customized(image, f, mode = 'bilinear')

        image_cropped = image_cropped.permute(0,2,3,1)
        image_cropped = image_cropped.reshape(b, b_pixels , output_size, output_size,c)
        image_cropped = image_cropped.reshape(b* b_pixels , output_size, output_size,c)
        image_cropped = image_cropped.permute(0,3,1,2)

        return image_cropped
    

    def forward(self, coordinate, x, return_cbp = False, factor = 1):

        b , n, n_angles = x.shape

        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n))))
        pad_width = (0,0,0, projection_size_padded - n)

        padded_sinogram = F.pad(x, pad_width)
        projection = torch.fft.fft(padded_sinogram, dim=1) * self.fourier_filter
        filtered_sinogram = torch.fft.ifft(projection, dim=1)[:,:n].real

        col = filtered_sinogram.permute(0,2,1).unsqueeze(1)
        t = self.t.repeat(b,1,1,1)

        cbp = F.grid_sample(col, t, align_corners= True)
        cbp = cbp.reshape(b, n_angles, config.image_size,config.image_size)

        if return_cbp:
            return cbp
        
        # if factor>1:
        #     # factor = np.random.uniform(1,4,size = 1)
        #     cbp = F.interpolate(cbp, size = int(config.image_size * factor), mode = 'bicubic')


        b , b_pixels , _ = coordinate.shape

        x = self.cropper(cbp , coordinate , output_size = config.w_size)
        mid_pix = torch.mean(x[:,:, config.w_size//2, config.w_size//2], dim = 1, keepdim=True) * np.pi/2 # FBP recon
        # x[:,:, config.w_size//2, config.w_size//2] = 0


        x = torch.flatten(x, 1)
        # coordinate = coordinate.reshape(b*b_pixels, -1)
        # x = torch.cat((coordinate, x), axis = 1)

        for i in range(len(self.MLP)-1):
            x = F.relu(self.MLP[i](x))

        x = self.MLP[-1](x)
        x = x + self.alpha * mid_pix # external skip connection to the centric pixel
        x = x.reshape(b, b_pixels, -1)
        
        return x




# class Bayesian_FunkNN(nn.Module):
#     '''Bayesian FunkNN module'''

#     def __init__(self, c):
#         super(Bayesian_FunkNN, self).__init__()
        
#         self.c = c
#         prev_unit = config.w_size**2 * self.c + 2
      
#         fcs = []
#         # prev_unit = 15*64
#         hidden_units = [10,10,10,10,9,9,9,9,8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,0]
#         hidden_units = np.power(2, hidden_units)
#         for i in range(len(hidden_units)):

#             # if i > 7 and i < 14:
#             # if i < 3:
#             if i > -1:
#                 fcs.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
#                                            in_features=prev_unit,
#                                            out_features=hidden_units[i],
#                                            sigma_init= 1e-2))
#             else:
#                 fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
                
#             prev_unit = hidden_units[i]

#         self.MLP = nn.ModuleList(fcs)
#         print(self.MLP[0].log_sigma_init)



#         # Adaptive receptive field
#         ws1 = torch.ones(1)
#         self.ws1 = nn.Parameter(ws1.clone().detach(), requires_grad=True)
#         ws2 = torch.ones(1)
#         self.ws2 = nn.Parameter(ws2.clone().detach(), requires_grad=True)

#         # Skip connection weight
#         alpha = torch.ones(1)
#         self.alpha = nn.Parameter(alpha.clone().detach(), requires_grad=True)
        
    
#     def grid_sample_customized(self, image, grid, mode = 'cubic_conv' , pad = 'reflect', align_corners = False):
#         '''Differentiable grid_sample:
#         equivalent performance with torch.nn.functional.grid_sample can be obtained by setting
#         align_corners = True,
#         pad: 'border': use border pixels,
#         'reflect': create reflect pad manually.
#         image is a tensor of shape (N, C, IH, IW)
#         grid is a tensor of shape (N, H, W, 2)'''

#         if mode == 'bilinear':
#             N, C, IH, IW = image.shape
#             _, H, W, _ = grid.shape

#             ix = grid[..., 0]
#             iy = grid[..., 1]


#             if align_corners == True:
#                 ix = ((ix + 1) / 2) * (IW-1);
#                 iy = ((iy + 1) / 2) * (IH-1);
                
#                 boundary_x = (0, IW-1)
#                 boundary_y = (0, IH-1)
                
            
#             elif align_corners == False:
#                 # ix = ((1+ix)*IW/2) - 1/2
#                 # iy = ((1+iy)*IH/2) - 1/2
                
#                 # boundary_x = (-1/2, IW-1/2)
#                 # boundary_y = (-1/2, IH-1/2)

#                 ix = ((1+ix)*IW/2) 
#                 iy = ((1+iy)*IH/2)
                
#                 boundary_x = (0, IW-1)
#                 boundary_y = (0, IH-1)
            

#             with torch.no_grad():
#                 ix_nw = torch.floor(ix);
#                 iy_nw = torch.floor(iy);
#                 ix_ne = ix_nw + 1;
#                 iy_ne = iy_nw;
#                 ix_sw = ix_nw;
#                 iy_sw = iy_nw + 1;
#                 ix_se = ix_nw + 1;
#                 iy_se = iy_nw + 1;

#             nw = (ix_se - ix)    * (iy_se - iy)
#             ne = (ix    - ix_sw) * (iy_sw - iy)
#             sw = (ix_ne - ix)    * (iy    - iy_ne)
#             se = (ix    - ix_nw) * (iy    - iy_nw)


#             if pad == 'reflect' or 'reflection':
                
#                 ix_nw = reflect_coords(ix_nw, boundary_x[0], boundary_x[1])
#                 iy_nw = reflect_coords(iy_nw, boundary_y[0], boundary_y[1])

#                 ix_ne = reflect_coords(ix_ne, boundary_x[0], boundary_x[1])
#                 iy_ne = reflect_coords(iy_ne, boundary_y[0], boundary_y[1])

#                 ix_sw = reflect_coords(ix_sw, boundary_x[0], boundary_x[1])
#                 iy_sw = reflect_coords(iy_sw, boundary_y[0], boundary_y[1])

#                 ix_se = reflect_coords(ix_se, boundary_x[0], boundary_x[1])
#                 iy_se = reflect_coords(iy_se, boundary_y[0], boundary_y[1])


#             elif pad == 'border':

#                 with torch.no_grad():
#                     torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
#                     torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

#                     torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
#                     torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

#                     torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
#                     torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

#                     torch.clamp(ix_se, 0, IW-1, out=ix_se)
#                     torch.clamp(iy_se, 0, IH-1, out=iy_se)


#             image = image.reshape(N, C, IH * IW)

#             nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
#             ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
#             sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
#             se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

#             out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
#                     ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
#                     sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
#                     se_val.view(N, C, H, W) * se.view(N, 1, H, W))


#             return out_val


#         elif mode == 'cubic_conv':

#             N, C, IH, IW = image.shape
#             _, H, W, _ = grid.shape

#             ix = grid[..., 0]
#             iy = grid[..., 1]


#             if align_corners == True:
#                 ix = ((ix + 1) / 2) * (IW-1);
#                 iy = ((iy + 1) / 2) * (IH-1);
                
#                 boundary_x = (0, IW-1)
#                 boundary_y = (0, IH-1)
                
            
#             elif align_corners == False:
#                 ix = ((1+ix)*IW/2) - 1/2
#                 iy = ((1+iy)*IH/2) - 1/2
                
#                 boundary_x = (-1/2, IW-1/2)
#                 boundary_y = (-1/2, IH-1/2)
            
#             # indices = [-2,-1,0,1,2,3] # order 6
#             indices = [-1,0,1,2] # order 4
#             # indices = [0,1] # order 2
#             points = cubic_coords(ix,iy, indices)
#             n_neighrbours = len(indices)**2

#             ix_relative = ix.unsqueeze(dim = -1)- points[...,0]
#             iy_relative = iy.unsqueeze(dim = -1)- points[...,1]

#             points[...,0] = reflect_coords(points[...,0], boundary_x[0], boundary_x[1])
#             points[...,1] = reflect_coords(points[...,1], boundary_y[0], boundary_y[1])
#             points = points.unsqueeze(dim = 1).expand(-1,C,-1,-1,-1,-1)

#             image = image.reshape(N, C, IH * IW)
#             points_values = torch.gather(image,2, (points[...,1] * IW + points[...,0]).long().view(N,C,H*W*n_neighrbours))
#             points_values = points_values.reshape(N,C,H,W,n_neighrbours)

#             ux = cubic_kernel(ix_relative, order = len(indices)).unsqueeze(dim = 1)
#             uy = cubic_kernel(iy_relative, order = len(indices)).unsqueeze(dim = 1)

#             recons = points_values * ux * uy
#             out_val = torch.sum(recons, dim = 4)

#             return out_val
        
    
#     def cropper(self, image, coordinate , output_size):
#         '''Cropper using Spatial Transformer'''
#         # Coordinate shape: b X b_pixels X 2
#         # image shape: b X c X h X w
#         d_coordinate = coordinate * 2
#         b , c , h , w = image.shape
#         b_pixels = coordinate.shape[1]
#         crop_size = 2 * (output_size-1)/(h-1)
#         x_m_x = crop_size/2
#         x_p_x = d_coordinate[:,:,1]
#         y_m_y = crop_size/2
#         y_p_y = d_coordinate[:,:,0]
#         theta = torch.zeros(b, b_pixels, 2,3).to(image.device)
#         theta[:,:,0,0] = x_m_x * self.ws1
#         theta[:,:,0,2] = x_p_x
#         theta[:,:,1,1] = y_m_y * self.ws2
#         theta[:,:,1,2] = y_p_y

#         theta = theta.reshape(b*b_pixels , 2 , 3)

#         f = F.affine_grid(theta, size=(b * b_pixels, c, output_size, output_size), align_corners=True)
#         f = f.reshape(b, b_pixels , output_size, output_size,2)
#         f = f.reshape(b, b_pixels * output_size, output_size,2)

#         # Non-differentiable grid_sampler
#         image_cropped = F.grid_sample(image, f, mode = 'bicubic', align_corners=True, padding_mode='reflection')
#         # Differentiable grid_sampler, well-suited for solving PDEs
#         # image_cropped = self.grid_sample_customized(image, f, mode = 'bilinear')

#         image_cropped = image_cropped.permute(0,2,3,1)
#         image_cropped = image_cropped.reshape(b, b_pixels , output_size, output_size,c)
#         image_cropped = image_cropped.reshape(b* b_pixels , output_size, output_size,c)
#         image_cropped = image_cropped.permute(0,3,1,2)

#         return image_cropped
    

#     def forward(self, coordinate, x):
#         b , b_pixels , _ = coordinate.shape

#         x = self.cropper(x , coordinate , output_size = config.w_size)
#         mid_pix = torch.mean(x[:,:, config.w_size//2, config.w_size//2], dim = 1, keepdim=True) * np.pi/2 # FBP recon

#         # x = x.permute(0,2,1,3)
#         # for i in range(len(self.CNNs)):

#         #     x = self.CNNs[i](x)
#         #     x = F.relu(x)
#         #     if i % 2 == 1:
#         #         x = self.maxpool(x)
#             # else:
#             #     if not (i ==0 or i == len(self.CNNs)-1):
#             #         x = x+x_temp # Internal skip connection

        
#         # coordinate = coordinate[..., None , None].expand(-1, -1, w_size, w_size)
#         # x = torch.cat((coordinate, x), axis = 1)

#         x = torch.flatten(x, 1)
#         coordinate = coordinate.reshape(b*b_pixels, -1)
#         x = torch.cat((coordinate, x), axis = 1)

#         # x = F.relu(self.bfc1(x))
#         # x = F.relu(self.bfc2(x))
#         # x = F.relu(self.bfc3(x))
#         # x = F.relu(self.bfc4(x))

#         for i in range(len(self.MLP)-1):
#             x = F.relu(self.MLP[i](x))

#         x = self.MLP[-1](x)
#         x = x + self.alpha * mid_pix # external skip connection to the centric pixel
#         x = x.reshape(b, b_pixels, -1)

#         return x


