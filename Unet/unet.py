import torch
import numpy as np
import os
import sys
sys.path.append('../')
from data_loader import *
from time import time
from utils import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def evaluation(i, subset, test_loader, model, results_path, noise_snr):

    print(f'Evaluation on {subset} set with {noise_snr}dB noise:')
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write(f'Evaluation on {subset} set with {noise_snr}dB noise:')
        file.write('\n')

    if subset == 'test':
        num_samples_write = 25
        ngrid = 5
    else:
        num_samples_write = 16
        ngrid = 4

    psnr_unet = 0
    psnr_fbp = 0
    ssim_unet = 0
    ssim_fbp = 0
    cnt = 0
    cmap = 'gray'

    for x_test,y_test in test_loader:

        if cnt > 0:
            break

        x_test = x_test.to(device)
        y_test = y_test.to(device)
        xhat_test = model(y_test)

        xhat_test = xhat_test.cpu().detach().numpy()
        x_test = x_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()

        psnr_unet += PSNR(x_test[:,0], xhat_test[:,0])
        psnr_fbp += PSNR(x_test[:,0], y_test[:,0])
        ssim_unet += SSIM(x_test[:,0], xhat_test[:,0])
        ssim_fbp += SSIM(x_test[:,0], y_test[:,0])
        
        cnt+=1

        # GT:
        images_np = x_test[:,0,:,:]
        image_write = images_np[:num_samples_write].reshape(
            ngrid, ngrid, image_size, image_size,1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)

        plt.imsave(os.path.join(results_path, f'{i}_{subset}_gt.png'),
            image_write[:,:,0], cmap = cmap)
            
        # FBP:
        y_np = y_test[:,0,:,:]
        y_write = y_np[:num_samples_write].reshape(
            ngrid, ngrid,image_size, image_size,1).swapaxes(1,2).reshape(ngrid*image_size, -1, 1)

        plt.imsave(os.path.join(results_path, f'{i}_{subset}_{noise_snr}_fbp.png'),
            y_write[:,:,0], cmap = cmap)
            
        
        # Recon:
        recon_np = xhat_test[:,0,:,:]
        recon_write = recon_np[:num_samples_write].reshape(
            ngrid, ngrid,image_size, image_size,1).swapaxes(1,2).reshape(ngrid*image_size, -1, 1)

        plt.imsave(os.path.join(results_path, f'{i}_{subset}_{noise_snr}_recon.png'),
            recon_write[:,:,0], cmap = cmap)
        
        if subset == 'ood' and i == -1:
            np.savez(os.path.join(results_path, f'{i}_{subset}_{noise_snr}_recon.npz'),
                     fbp = y_write[:,:,0], unet = recon_write[:,:,0]) 

    psnr_unet /= cnt
    psnr_fbp /= cnt
    ssim_unet /= cnt
    ssim_fbp /= cnt

    np.savez(os.path.join(exp_path, f'unet_{subset}.npz'),
        images = images_np, fbp = y_np, unet = recon_np)

    print('unet_PSNR : {:.1f}| fbp_PSNR : {:.1f} | unet_SSIM : {:.2f}| fbp_SSIM : {:.2f}'.format(psnr_unet, psnr_fbp,
            ssim_unet, ssim_fbp))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('unet_PSNR : {:.1f}| fbp_PSNR : {:.1f} | unet_SSIM : {:.2f}| fbp_SSIM : {:.2f}'.format(psnr_unet, psnr_fbp,
            ssim_unet, ssim_fbp))
        file.write('\n')


# Params
ood_analysis = True
N_epochs = 200
batch_size = 64
image_size = 128
num_angles = 30
gpu_num = 2
run_train = True
exp_path = 'experiments/uncalibrated/'
myloss = F.mse_loss
# myloss = F.l1_loss
# train_path = '../../../datasets/CT/original_data/train'
# test_path = '../../../datasets/CT/original_data/test'
# ood_path = '../../datasets/CT_brain/test_samples/images'
train_path = '../datasets/128_30_complete_30_right/train'
test_path = '../datasets/128_30_complete_30_right/test'
ood_path = '../datasets/128_30_complete_30_right/outlier'
unet_reload = True
test_noise_snr = 30
ood_noise_snr = 30
train_noise_snr = 30

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)

results_path = exp_path + 'results/'
if os.path.exists(results_path) == False:
    os.mkdir(results_path)

device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=1, init_features=32, pretrained=False).to(device)

# model = nn.DataParallel(model)

num_param = count_parameters(model)
print('---> Number of trainable parameters of supercnn: {}'.format(num_param))


# Dataset:
train_dataset = CT_dataset(train_path, unet = True, train = True)
test_dataset = CT_dataset(test_path, unet = True, train = False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=24)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)

n_ood = 0
if ood_analysis:
    ood_dataset = CT_dataset(ood_path, unet = True, train = False)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, num_workers=24)
    n_ood= len(ood_loader.dataset)

print('---> Number of training, test and ood samples: {}, {}, {}'.format(ntrain,n_test, n_ood))
plot_per_num_epoch = 1 if ntrain > 10000 else 30000//ntrain

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint_path = exp_path + '/model.pt'
if os.path.exists(checkpoint_path) and unet_reload:
    checkpoint_unet = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_unet['model_state_dict'])
    optimizer.load_state_dict(checkpoint_unet['optimizer_state_dict'])
    print('Unet is restored...')


if run_train:

    loss_funknn_plot = np.zeros([N_epochs//plot_per_num_epoch])
    for i in range(N_epochs):
        start = time()
        loss_unet_epoch = 0
        for x,y in train_loader:

            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            xhat = model(y)
            loss = myloss(x.reshape(batch_size, -1) , xhat.reshape(batch_size, -1) )
            loss.backward()
            optimizer.step()
            loss_unet_epoch += loss.item()

        if i % plot_per_num_epoch == 0 or (i + 1) == N_epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            
            end = time()
            loss_unet_epoch/= ntrain
            i_th = i // plot_per_num_epoch
            loss_funknn_plot[i_th] = loss_unet_epoch
            
            plt.plot(np.arange(N_epochs)[:i_th] , loss_funknn_plot[:i_th], 'o-', linewidth=2)
            plt.title('UNet_loss')
            plt.xlabel('epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(exp_path, 'unet_loss.jpg'))
            np.save(os.path.join(exp_path, 'unet_loss.npy'), loss_funknn_plot[:i_th])
            plt.close()

            print('epoch: {}/{} | time: {:.0f} | UNet_loss: {:.6f} | gpu: {:.0f}'.format(i, N_epochs, end-start,
                                                                                    loss_unet_epoch, gpu_num))
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('ep: {}/{} | time: {:.0f} | FunkNN_loss: {:.6f} | gpu: {:.0f}'.format(i, N_epochs, end-start,
                                                                                    loss_unet_epoch, gpu_num))
                file.write('\n')

            evaluation(i, 'test', test_loader, model, results_path, test_noise_snr)
            if ood_analysis:
                evaluation(i, 'ood', ood_loader, model, results_path, ood_noise_snr)


evaluation(-1, 'test', test_loader, model, results_path, test_noise_snr)
if ood_analysis:
    evaluation(-1, 'ood', ood_loader, model, results_path, ood_noise_snr)

        