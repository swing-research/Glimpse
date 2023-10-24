import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import radon, iradon
from scipy import optimize
from skimage.metrics import peak_signal_noise_ratio as psnr
import config_funknn as config



def PSNR(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_true[i],
             x_pred[i],
             data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]



def PSNR_rescale(x_true , x_pred):
    '''Calculate SNR rescale of a batch of true and their estimations'''
    snr = 0
    for i in range(x_true.shape[0]):
        
        def func(weights):
            x_pred_rescale=  weights[0]*x_pred[i]+weights[1]
            s = psnr(x_pred_rescale,
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min())
            
            return s
        opt = optimize.minimize(lambda x: -func(x),x0=np.array([1,0]))
        snr += -opt.fun
        weights = opt.x
    return snr/x_true.shape[0], weights



def SNR(x_true , x_pred):
    '''Calculate SNR of a batch of true and their estimations'''

    snr = 0
    for i in range(x_true.shape[0]):
        Noise = x_true[i] - x_pred[i]
        Noise_power = np.sum(np.square(np.abs(Noise)))
        Signal_power = np.sum(np.square(np.abs(x_true[i])))
        snr += 10*np.log10(Signal_power/Noise_power)
  
    return snr/x_true.shape[0]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def batch_sampling(image_recon, coords, c, model):
    s = 512
    # model.freeze()
    for m in model.modules() :
        try:
            m.freeze()
        except:
            pass

    outs = np.zeros([np.shape(coords)[0], np.shape(coords)[1], c])
    for i in range(int(np.ceil(np.shape(coords)[1]/s))):
        
        batch_coords = coords[:,i*s: (i+1)*s]
        out = model(batch_coords, image_recon).detach().cpu().numpy()
        outs[:,i*s: (i+1)*s] = out
    
    for m in model.modules() :
        try:
            m.unfreeze()
        except:
            pass

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



def posterior_visualization(posterior_samples, gt , y, n_test = 5 , n_sample_show = 4):
    '''Generate posterior samples, MMSE, MAP and UQ'''
    
    def normalization(image):
        image += -image.min()
        image /= image.max()
        
        return image
    
    
    n_sample = n_sample_show + 5 
    output_shape = [n_test*n_sample, np.shape(posterior_samples)[2] , np.shape(posterior_samples)[3]]
    output = np.zeros(output_shape)

    
    for i in range(n_test):
        output[i*n_sample] = gt[i]
        output[i*n_sample + 1] = np.mean(posterior_samples[i] , axis = 0)
        output[i*n_sample+2:i*n_sample + 2 + n_sample_show] = posterior_samples[i, 0:n_sample_show]
        output[i*n_sample + 4 + n_sample_show] = y[i]
        output[i*n_sample + 2 + n_sample_show] = normalization(np.std(posterior_samples[i] , axis = 0))
        std_fft = np.std(np.log(np.absolute(np.fft.fftshift(np.fft.fft2(posterior_samples[i])))) , axis = 0)
        output[i*n_sample + 3 + n_sample_show] = normalization(std_fft)
        
    return output





