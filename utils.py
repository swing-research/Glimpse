import numpy as np
import torch
from skimage.transform import iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import config
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular


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



def batch_sampling(image_recon, coords, model, s = 512):

    
    outs = np.zeros([np.shape(coords)[0], np.shape(coords)[1]])
    with torch.no_grad():
        for i in range(int(np.ceil(np.shape(coords)[1]/s))):
            
            batch_coords = coords[:,i*s: (i+1)*s]
            out = model(batch_coords, image_recon).detach().cpu().numpy()
            outs[:,i*s: (i+1)*s] = out
    
    return outs


def get_mgrid(n1, n2, n3):
    # Generate 3D pixel coordinates from a volume of n1 x n2 x n3
    pixel_coords = np.stack(np.mgrid[:n1, :n2, :n3], axis=-1).astype(np.float32)
    scale = np.max([n1,n2,n3])
    pixel_coords[:,:,:,0] = 2*((pixel_coords[:,:,:,0]/(n1-1)) -0.5) * n1/scale
    pixel_coords[:,:,:,1] = 2*(pixel_coords[:,:,:,1]/(n2-1) -0.5) * n2/scale
    pixel_coords[:,:,:,2] = 2*(pixel_coords[:,:,:,2]/(n3-1) -0.5) * n3/scale
    pixel_coords = torch.Tensor(pixel_coords.copy())
    return pixel_coords



def fbp_batch(sinograms, theta):

    fbps = []
    for i in range(sinograms.shape[0]):
        fbps.append(iradon(sinograms[i], theta = theta, circle = False))

    fbps = np.array(fbps)
    return fbps


def generate_projections(vol,angles):
    # note angles in randians
    n1 = vol.shape[0]
    n2 = vol.shape[1]
    n3 = vol.shape[2]

    # arranging the volume to get the same projections as the matlab simulator
    # vol_swap = torch.zeros_like(vol)

    # for i in range(vol.shape[2]):
    #     vol_swap[:,:,i]  = vol[:,:,-i].clone()

    operator =  ParallelBeamGeometry3DOpAngles_rectangular((n1,n2,n3), angles, op_snr=np.inf, fact=1)
    proj = operator(vol)
    return proj


def noise_simulation(proj, noise_level):

    # Add noise
    # if noise level is a list, then sample from it
    if isinstance(noise_level, list):
        noise_level = np.random.uniform(low = min(noise_level),
                                            high = max(noise_level), size=1)[0]
    else:
        noise_level = noise_level
    # TODO : include Gaussian approximation of poisson noise
    sigma_value = find_sigma_noise(noise_level,proj)
    proj = proj + sigma_value*torch.randn_like(proj) #+  abs(proj)*torch.randn_like(proj)*alpha
    return proj


def generate_FBP(proj, angles, n1 = 800 , n2 = 800 , n3 = 300):

    operator =  ParallelBeamGeometry3DOpAngles_rectangular((n1,n2,n3), angles, op_snr=np.inf, fact=1)
    return  operator.pinv(proj)



def find_sigma_noise(SNR_value,x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)



def custom_ramp_fft(x,t_cust):
    """ 
    ramp filtering using torch fft 
    x: (n_projections, N , N) tensor
    t_cust: (2*N) tensor
    """
    # print(x.shape, t_cust.shape)
    projection_fft = torch.fft.fftn(x, dim=(-1), s = (2*x.shape[-1]))
    projection_fft = projection_fft*t_cust[None,None,None,...]
    projection_filtered = torch.fft.ifftn(projection_fft, dim=(-1), s = -1) #(2*x.shape[-1]))
    projection_filtered = projection_filtered[:,:,:,0:x.shape[-1]].real
    # print(projection_filtered.shape)
    return projection_filtered
