import numpy as np
import torch
import os
from utils import *
import config
import matplotlib.pyplot as plt
# from simulator import Simulator
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular



def evaluator(ep, subset, data_loader, model, exp_path, operator):

    # projection_simulator = Simulator(config.data)
    # operator =  ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3),
    #                                                    config.data.angles, op_snr=np.inf,
    #                                                    fact=1)

    results_folder = os.path.join(exp_path, 'Results')
    os.makedirs(results_folder, exist_ok= True)

    device = model.z.device


    print(f'Evaluation on {subset} set:')
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write(f'Evaluation on {subset} set:')
        file.write('\n')

    vol = next(iter(data_loader))

    vol = vol.to(device)
    proj = operator(vol)
    proj = noise_simulation(proj, config.data.noise_level)

    print(vol.shape, vol.min(), vol.max(), vol.mean(), torch.var(vol))
    print(proj.shape, proj.min(), proj.max(), proj.mean(), torch.var(proj))


    # GT:
    vol_np = vol.detach().cpu().numpy()[0,:,:,150]# [0,:,60,:]# [0,:,:,10] ##
    print(vol_np.shape, vol_np.min(), vol_np.max(), vol_np.mean())
    
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_gt.png'),
        vol_np, cmap = config.cmap)
    

    # FBP:
    fbp = operator.pinv(proj[0]).detach().cpu().numpy()[:,:,150]#[:,60,:]#
    print(fbp.shape, fbp.min(), fbp.max(), fbp.mean())
    
    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_fbp.png'),
               fbp, cmap = config.cmap)


    # Glimpse:
    model.eval()
    coords = get_mgrid(config.n1, config.n2, config.n3)
    coords = coords[:,:,150:151]
    # coords = coords[:,60:61,:]
    coords = coords.reshape(-1, 3)
    coords = torch.unsqueeze(coords, dim = 0).to(device)
    recon_np = batch_sampling(proj, coords,model, s = 4000)
    recon_np = np.reshape(recon_np, [config.n1, -1])
    print(recon_np.shape, recon_np.min(), recon_np.max(), recon_np.mean())

    plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_glimpse.png'),
        recon_np, cmap = config.cmap)  
    

    # # Eror
    # error = np.abs(images_np - recon_np)
    # error_write = error[:num_samples_write].reshape(
    #     ngrid, ngrid, config.image_size, config.image_size, 1).swapaxes(1, 2).reshape(ngrid*config.image_size, -1, 1)

    # plt.imsave(os.path.join(results_folder, f'{ep}_{subset}_error.png'),
    #     error_write[:,:,0], cmap = 'seismic')

    
    # Numerics
    psnr_recon = PSNR(vol_np, recon_np)
    psnr_fbp = PSNR(vol_np, fbp)
    ssim_recon = 0 #SSIM(vol_np, recon_np)
    ssim_fbp = 0# SSIM(vol_np, fbp)

    # np.savez(os.path.join(exp_path, f'glimpse_{subset}.npz'),
    #         images = images_np, fbp = fbp, glimpse = recon_np)

    print('PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
        psnr_fbp, psnr_recon, ssim_fbp, ssim_recon))

    # with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
    #     file.write('PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
    #         psnr_fbp, psnr_recon, ssim_fbp, ssim_recon))
    #     file.write('\n')
    #     if subset == 'ood':
    #         file.write('\n')





def evaluator_demo(data_loader, model, image_size, theta_init, cmap):

    device = model.ws1.device
    num_samples_write = 1
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    images, sinogram = next(iter(data_loader))
    images = images.to(device)
    sinogram = sinogram.to(device)
    images = images.reshape(-1, image_size, image_size, 1)


    # GT:
    images_np = images.detach().cpu().numpy()[:,:,:,0]
    image_write = images_np[:num_samples_write].reshape(
        ngrid, ngrid,
        image_size, image_size,1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)


    # FBP:
    sinogram_np = sinogram.detach().cpu().numpy()
    fbp = fbp_batch(sinogram_np, theta = theta_init)
    fbp_write = fbp[:num_samples_write].reshape(
        ngrid, ngrid,
        image_size, image_size,1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)


    # Glimpse:
    with torch.no_grad():
        coords = get_mgrid(image_size)
        coords = torch.unsqueeze(coords, dim = 0)
        coords = coords.expand(images.shape[0] , -1, -1).to(device)
        recon_np = batch_sampling(sinogram, coords,1, model)

    recon_np = np.reshape(recon_np, [-1, image_size, image_size,1])[:,:,:,0]
    recon_write = recon_np[:num_samples_write].reshape(
        ngrid, ngrid, image_size, image_size, 1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)



    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(fbp_write[:,:,0], cmap = cmap); plt.title('FBP')
    plt.subplot(1,3,2); plt.imshow(recon_write[:,:,0], cmap = cmap); plt.title('Glimpse')
    plt.subplot(1,3,3); plt.imshow(image_write[:,:,0], cmap = cmap); plt.title('GT')
    plt.show()

    # Numerics
    psnr_recon = PSNR(images_np, recon_np)
    psnr_fbp = PSNR(images_np, fbp)
    ssim_recon = SSIM(images_np, recon_np)
    ssim_fbp = SSIM(images_np, fbp)

    print('PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
        psnr_fbp, psnr_recon, ssim_fbp, ssim_recon))




