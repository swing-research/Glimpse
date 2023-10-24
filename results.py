import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
import config_funknn as config
import matplotlib.pyplot as plt



def evaluator_sinogram(ep, subset, data_loader, model, exp_path):

    results_folder = os.path.join(exp_path, 'Results')
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    noise_snr = config.test_noise_snr
    device = model.ws1.device
    num_samples_write = config.sample_number if config.sample_number < 26 else 25
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    if subset == 'ood':
        num_samples_write = 16
        ngrid = 4
        noise_snr = config.ood_noise_snr

    print('Evaluation on {} set with {}dB noise:'.format(subset, noise_snr))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('Evaluation on {} set with {}dB noise:'.format(subset, noise_snr))
        file.write('\n')


    images, sinogram = next(iter(data_loader))

    images = images[:config.sample_number].to(device)
    sinogram = sinogram[:config.sample_number].to(device)
    images = images.reshape(-1, config.image_size, config.image_size, 1)


    # GT:
    images_np = images.detach().cpu().numpy()[:,:,:,0]
    image_write = images_np[:num_samples_write].reshape(
        ngrid, ngrid,
        config.image_size, config.image_size,1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_gt.png'),
        image_write[:,:,0], cmap = config.cmap)
    

    # FBP:
    sinogram_np = sinogram.detach().cpu().numpy()
    fbp = fbp_batch(sinogram_np)
    fbp_write = fbp[:num_samples_write].reshape(
        ngrid, ngrid,
        config.image_size, config.image_size,1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_{noise_snr}db_fbp.png'),
               fbp_write[:,:,0], cmap = config.cmap)



    model.eval()
    # Recon:
    coords = get_mgrid(config.image_size)
    coords = torch.unsqueeze(coords, dim = 0)
    coords = coords.expand(images.shape[0] , -1, -1).to(device)
    recon_np = batch_sampling(sinogram, coords,1, model)
    recon_np = np.reshape(recon_np, [-1, config.image_size, config.image_size,1])[:,:,:,0]

    recon_write = recon_np[:num_samples_write].reshape(
        ngrid, ngrid, config.image_size, config.image_size, 1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_{noise_snr}db_deep_local.png'),
        recon_write[:,:,0], cmap = config.cmap)  


    # Eror
    error = np.abs(images_np - recon_np)
    error_write = error[:num_samples_write].reshape(
        ngrid, ngrid, config.image_size, config.image_size, 1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)

    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_{noise_snr}db_error.png'),
        error_write[:,:,0], cmap = 'seismic')

    
    # Numerics
    psnr_recon = PSNR(images_np, recon_np)
    psnr_fbp = PSNR(images_np, fbp)

    print('PSNR fbp: {:.1f} | PSNR Deep_local: {:.1f}'.format(
        psnr_fbp, psnr_recon))

    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('PSNR fbp: {:.1f} | PSNR Deep_local: {:.1f}'.format(
        psnr_fbp, psnr_recon))
        file.write('\n')
        if subset == 'ood':
            file.write('\n')

