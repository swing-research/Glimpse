import numpy as np
import torch
import os
from utils import *
import config
import matplotlib.pyplot as plt



def evaluator(ep, subset, data_loader, model, exp_path):

    results_folder = os.path.join(exp_path, 'Results')
    os.makedirs(results_folder, exist_ok= True)

    device = model.ws1.device
    num_samples_write = config.sample_number if config.sample_number < 26 else 25
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    if subset == 'ood':
        num_samples_write = 16
        ngrid = 4

    print(f'Evaluation on {subset} set:')
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write(f'Evaluation on {subset} set:')
        file.write('\n')

    images, sinogram = next(iter(data_loader))

    images = images.to(device)
    sinogram = sinogram.to(device)
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
    fbp = fbp_batch(sinogram_np, theta = config.theta_init)
    fbp_write = fbp[:num_samples_write].reshape(
        ngrid, ngrid,
        config.image_size, config.image_size,1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_fbp.png'),
               fbp_write[:,:,0], cmap = config.cmap)


    # Glimpse:
    model.eval()
    coords = get_mgrid(config.image_size)
    coords = torch.unsqueeze(coords, dim = 0)
    coords = coords.expand(images.shape[0] , -1, -1).to(device)
    recon_np = batch_sampling(sinogram, coords,1, model)
    recon_np = np.reshape(recon_np, [-1, config.image_size, config.image_size,1])[:,:,:,0]

    recon_write = recon_np[:num_samples_write].reshape(
        ngrid, ngrid, config.image_size, config.image_size, 1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_glimpse.png'),
        recon_write[:,:,0], cmap = config.cmap)  


    # Eror
    error = np.abs(images_np - recon_np)
    error_write = error[:num_samples_write].reshape(
        ngrid, ngrid, config.image_size, config.image_size, 1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)

    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_error.png'),
        error_write[:,:,0], cmap = 'seismic')

    
    # Numerics
    psnr_recon = PSNR(images_np, recon_np)
    psnr_fbp = PSNR(images_np, fbp)
    ssim_recon = SSIM(images_np, recon_np)
    ssim_fbp = SSIM(images_np, fbp)

    np.savez(os.path.join(exp_path, f'glimpse_{subset}.npz'),
            images = images_np, fbp = fbp, glimpse = recon_np)

    print('PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
        psnr_fbp, psnr_recon, ssim_fbp, ssim_recon))

    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
            psnr_fbp, psnr_recon, ssim_fbp, ssim_recon))
        file.write('\n')
        if subset == 'ood':
            file.write('\n')




