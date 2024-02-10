import numpy as np
import torch
from skimage.transform import iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import config


def SSIM(x_true , x_pred):
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min(),
                  channel_axis = False)
        
    return s/np.shape(x_pred)[0]




def PSNR(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_true[i],
             x_pred[i],
             data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def batch_sampling(image_recon, coords, c, model, s = 512):

    
    outs = np.zeros([np.shape(coords)[0], np.shape(coords)[1], c])
    with torch.no_grad():
        for i in range(int(np.ceil(np.shape(coords)[1]/s))):
            
            batch_coords = coords[:,i*s: (i+1)*s]
            out = model(batch_coords, image_recon).detach().cpu().numpy()
            outs[:,i*s: (i+1)*s] = out
    
    return outs


def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1).astype(np.float32)
    pixel_coords /= (sidelen-1)   
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).reshape(-1, 2)
    return pixel_coords



def fbp_batch(sinograms):

    fbps = []
    for i in range(sinograms.shape[0]):
        fbps.append(iradon(sinograms[i], theta=config.theta, circle = False))

    fbps = np.array(fbps)
    return fbps

