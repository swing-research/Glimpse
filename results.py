import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
import config_funknn as config
import matplotlib.pyplot as plt



def evaluator_sinogram(ep, subset, data_loader, model, exp_path):

    samples_folder = os.path.join(exp_path, 'Results')
    if not os.path.exists(samples_folder):
        os.mkdir(samples_folder)
    image_path_reconstructions = os.path.join(
        samples_folder, 'Reconstructions')

    if not os.path.exists(image_path_reconstructions):
        os.mkdir(image_path_reconstructions)

    max_scale = config.max_scale
    sample_number = config.sample_number
    image_size = config.image_size
    cmap = config.cmap
    noise_snr = config.test_noise_snr
    device = model.ws1.device
    num_samples_write = sample_number if sample_number < 26 else 25
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    if subset == 'ood':
        max_scale = 1
        num_samples_write = 16
        ngrid = 4
        noise_snr = config.ood_noise_snr

    print('Evaluation on {} set with {}dB noise:'.format(subset, noise_snr))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('Evaluation on {} set with {}dB noise:'.format(subset, noise_snr))
        file.write('\n')

    if config.max_scale > 1 and subset == 'test':
        images, sinogram, images_x2 = next(iter(data_loader))
        images_x2 = images_x2.reshape(-1, 2*image_size, 2*image_size, 1)

        # GTx2:
        images_x2_np = images_x2.detach().cpu().numpy()[:,:,:,0]
        images_x2_write = images_x2_np[:num_samples_write].reshape(
            ngrid, ngrid,
            2*config.image_size, 2*config.image_size,1).swapaxes(1, 2).reshape(ngrid*2*config.image_size, -1, 1)
        
        plt.imsave(os.path.join(image_path_reconstructions, f'{ep}_{subset}_gt2.png'),
            images_x2_write[:,:,0], cmap = cmap)
        
    else:
        images, sinogram = next(iter(data_loader))
    images = images[:sample_number].to(device)

    sinogram = sinogram[:sample_number].to(device)
    images = images.reshape(-1, image_size, image_size, 1)


    # GT:
    images_np = images.detach().cpu().numpy()[:,:,:,0]
    image_write = images_np[:num_samples_write].reshape(
        ngrid, ngrid,
        config.image_size, config.image_size,1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    
    plt.imsave(os.path.join(image_path_reconstructions, f'{ep}_{subset}_gt.png'),
        image_write[:,:,0], cmap = cmap)
    

    # FBP:
    cbp = model(0, sinogram, return_cbp = True)
    cbp_np = cbp.detach().cpu().numpy().mean(axis = 1) * np.pi/2
    cbp_write = cbp_np[:num_samples_write].reshape(
        ngrid, ngrid,
        config.image_size, config.image_size,1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)
    
    plt.imsave(os.path.join(image_path_reconstructions, f'{ep}_{subset}_{noise_snr}db_fbp.png'),
               cbp_write[:,:,0], cmap = cmap)


    scales = [i for i in range(int(np.log2(2*max_scale)))]
    scales = np.power(2, scales)

    for i in range(len(scales)):
        res = scales[i]*image_size
        factor = 1

        if i > 0:
            
            recon_bicubic = F.interpolate(recon, size = res, antialias = True, mode = 'bicubic')
            recon_bicubic = recon_bicubic.detach().cpu().numpy()[:,0]
            recon_bicubic_write = recon_bicubic[:num_samples_write].reshape(
                ngrid, ngrid, res, res, 1).swapaxes(1, 2).reshape(ngrid*res, -1, 1)

            plt.imsave(os.path.join(image_path_reconstructions, f'{ep}_{subset}_{noise_snr}db_bicubic_{scales[i]}.png'),
                recon_bicubic_write[:,:,0], cmap = cmap)
            factor = 2

        # Recon:
        coords = get_mgrid(res).reshape(-1, 2)
        coords = torch.unsqueeze(coords, dim = 0)
        coords = coords.expand(images.shape[0] , -1, -1).to(device)
        print(factor)
        recon_np = batch_sampling(sinogram, coords,1, model, factor)
        recon_np = np.reshape(recon_np, [-1, res, res])
        recon_write = recon_np[:num_samples_write].reshape(
            ngrid, ngrid, res, res, 1).swapaxes(1, 2).reshape(ngrid*res, -1, 1)

        plt.imsave(os.path.join(image_path_reconstructions, f'{ep}_{subset}_{noise_snr}db_recon_{scales[i]}.png'),
            recon_write[:,:,0], cmap = cmap)      

        if subset == 'ood' and ep == -1:
            np.savez(os.path.join(image_path_reconstructions, f'{ep}_{subset}_{noise_snr}db_recon_{scales[i]}.npz'),
                     gt = image_write[:,:,0], funknn = recon_write[:,:,0]) 
        
        if i == 0:
            psnr_recon = PSNR(images_np, recon_np)
            psnr_cbp = PSNR(images_np, cbp_np)

            print('PSNR_fbp_f{}: {:.1f} | PSNR_recon_f{}: {:.1f}'.format(scales[i],
                psnr_cbp, scales[i], psnr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('PSNR_fbp_f{}: {:.1f} | PSNR_recon_f{}: {:.1f} | '.format(scales[i],
                psnr_cbp, scales[i], psnr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')

            recon = torch.tensor(recon_np, dtype = torch.float32).unsqueeze(1)

        if i == 1:
            psnr_recon = PSNR(images_x2_np, recon_np)
            psnr_bicubic = PSNR(images_x2_np, recon_bicubic)

            print('PSNR_bicubic_f{}: {:.1f} | PSNR_recon_f{}: {:.1f}'.format(scales[i],
                psnr_bicubic, scales[i], psnr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('PSNR_bicubic_f{}: {:.1f} | PSNR_recon_f{}: {:.1f}'.format(scales[i],
                psnr_bicubic, scales[i], psnr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')






def evaluator_Bayesian(ep, subset, data_loader, model, exp_path):

    samples_folder = os.path.join(exp_path, 'Results')
    if not os.path.exists(samples_folder):
        os.mkdir(samples_folder)
    image_path_reconstructions = os.path.join(
        samples_folder, 'Reconstructions')

    if not os.path.exists(image_path_reconstructions):
        os.mkdir(image_path_reconstructions)

    max_scale = config.max_scale
    sample_number = config.sample_number
    image_size = config.image_size
    cmap = config.cmap
    noise_snr = config.test_noise_snr
    device = model.ws1.device
    num_samples_write = sample_number if sample_number < 26 else 25
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    if subset == 'ood':
        max_scale = 2
        noise_snr = config.ood_noise_snr

    print('Evaluation on {} set with {}dB noise:'.format(subset, noise_snr))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('Evaluation over {} set:'.format(subset))
        file.write('\n')

    images, sinogram = next(iter(data_loader))
    images = images[:sample_number].to(device)
    sinogram = sinogram[:sample_number].to(device)
    images = images.reshape(-1, image_size, image_size, 1)

    # GT:
    images_np = images.detach().cpu().numpy()[:,:,:,0]

    # FBP:
    cbp = model(0, sinogram, return_cbp = True)
    fbp_np = cbp.detach().cpu().numpy().mean(axis = 1) * np.pi/2

    scales = [i for i in range(int(np.log2(2*max_scale)))]
    scales = np.power(2, scales)


    for i in range(len(scales)):
        res = scales[i]*image_size
        posterior_samples = np.zeros([config.num_posteriors, images.shape[0], res, res])
        for j in range(config.num_posteriors):

            # Recon:
            coords = get_mgrid(res).reshape(-1, 2)
            coords = torch.unsqueeze(coords, dim = 0)
            coords = coords.expand(images.shape[0] , -1, -1).to(device)
            recon = batch_sampling(sinogram, coords,1, model)
            posterior_samples[j] = np.reshape(recon, [-1, res, res])

        posterior_samples = posterior_samples.transpose(1,0,2,3)
        posterir_write = posterior_visualization(posterior_samples, images_np , fbp_np,
                                n_test = config.sample_number, n_sample_show = 4)
        posterir_write = posterir_write.reshape(
            num_samples_write, 4 + 5, res, res, 1).swapaxes(1, 2).reshape(num_samples_write*res, -1, 1)

        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_funknn_%d.png' % (ep,scales[i])),
            posterir_write[:,:,0], cmap = cmap)              
        
        if i == 0:
            mmse = np.mean(posterior_samples, axis = 1)
            psnr_recon = PSNR(images_np, mmse)
            psnr_fbp = PSNR(images_np, fbp_np)

            print('PSNR_fbp_f{}: {:.1f} | PSNR_mmse_f{}: {:.1f}'.format(scales[i],
                                                                      psnr_fbp, scales[i], psnr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('PSNR_fbp_f{}: {:.1f} | PSNR_mmse_f{}: {:.1f} | '.format(scales[i],
                                                                                  psnr_fbp, scales[i], psnr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')

