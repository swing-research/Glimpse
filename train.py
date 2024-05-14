import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from glimpse import glimpse
from utils import *
from data_loader import *
from results import evaluator
import config
# from simulator import Simulator
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular

torch.manual_seed(0)
np.random.seed(0)


enable_cuda = True
device = torch.device('cuda:' + str(config.gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

# experiment path
exp_path = 'experiments_3D/' + config.exp_desc
os.makedirs(exp_path, exist_ok=True)


step_size = 50
gamma = 0.5
myloss = F.mse_loss
# myloss = F.l1_loss
num_batch_pixels = 100 # The number of iterations over each batch
batch_pixels = 3000 # Number of pixels to optimize in each iteration

# Print the experiment setup:
print('Experiment setup:')
print(f'---> num epochs: {config.n_epochs}')
print(f'---> batch_size: {config.batch_size}')
print(f'---> Learning rate: {config.learning_rate}')
print(f'---> experiment path: {exp_path}')

# Dataset:


train_dataset = simulatorVolumes(root_dir= config.train_path, normalize_type= 'standard',
                                 models= config.train_samples, n1 = config.n1)
test_dataset = simulatorVolumes(root_dir= config.test_path, normalize_type= 'standard',
                                models= config.test_samples, n1 = config.n1)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=0,
                                           shuffle = True, pin_memory = True,pin_memory_device = 'cuda:' + str(config.gpu_num))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0,
                                           shuffle = False, pin_memory = True,pin_memory_device = 'cuda:' + str(config.gpu_num))

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)


print(f'---> Number of training, test and ood samples: {ntrain}, {n_test}')

# Loading model
plot_per_num_epoch = 1

model = glimpse(n1 = config.n1, n2 = config.n2 , n3 = config.n3,
                patch_shape= config.patch_shape, learned_geo= config.learned_geo,
                theta_init = config.data.angles, lsg = config.lsg,
                learnable_filter = config.learnable_filter,
                filter_init = config.filter_init).to(device)
# model = torch.nn.DataParallel(model) # Using multiple GPUs
num_param = count_parameters(model)
print('---> Number of trainable parameters: {}'.format(num_param))

optimizer = Adam(model.parameters(), lr=config.learning_rate)

checkpoint_exp_path = os.path.join(exp_path, 'glimpse.pt')
if os.path.exists(checkpoint_exp_path) and config.restore_model:
    checkpoint_glimpse = torch.load(checkpoint_exp_path)
    model.load_state_dict(checkpoint_glimpse['model_state_dict'])
    optimizer.load_state_dict(checkpoint_glimpse['optimizer_state_dict'])
    print('glimpse is restored...')

operator =  ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3),
                                                       config.data.angles, op_snr=np.inf,
                                                       fact=1)

operator_real =  ParallelBeamGeometry3DOpAngles_rectangular((1024,1024,300),
                                                       config.data.angles, op_snr=np.inf,
                                                       fact=1)

evaluator(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path,
          operator = operator, operator_real = operator_real)
if config.train:
    print('Training...')

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = config.n_epochs + 1 # only plot in the last epoch
    
    loss_plot = np.zeros([config.n_epochs])
    for ep in range(config.n_epochs):
        model.train()
        t1 = default_timer()
        loss_epoch = 0
        cnt = 0

        for vol in train_loader:
            
            vol = vol[0].to(device)

            proj = operator(vol)
            proj = noise_simulation(proj, config.data.noise_level)
            proj = proj[None,...].detach()

            vol = vol[...,None]
            vol = vol.reshape(-1,1)

            coords = get_mgrid(config.n1, config.n2, config.n3)
            coords = coords.reshape(-1, 3)
            vol = vol.cpu().numpy()
            coords = coords.cpu().numpy()

            pixel_data = pixel_loader(vol, coords, n_samples= num_batch_pixels * batch_pixels)
            pixel_dataset = torch.utils.data.DataLoader(pixel_data,
                                                       batch_size=batch_pixels,
                                                       shuffle = False,)
                                                    #    num_workers=16,
                                                    #    pin_memory = True,
                                                    #    pin_memory_device = 'cuda:' + str(config.gpu_num))
            
            del vol, coords


            # print(vol.shape, vol.min(), vol.max())
            # print(proj.shape, proj.min(), proj.max())

            for batch_coords, batch_image in pixel_dataset:

                batch_coords = batch_coords.to(device)[None, ...]
                batch_image = batch_image.to(device)[None,...]

                optimizer.zero_grad()

                out = model(batch_coords, proj)
                mse_loss = myloss(out.reshape(1, -1) , batch_image[:,:,0].reshape(1, -1) )
                total_loss = mse_loss

                total_loss.backward()
                optimizer.step()
                loss_epoch += total_loss.item()

                # Free up memory periodically
                torch.cuda.empty_cache()
            
            del pixel_dataset, pixel_data
            
            print(cnt)
            cnt = cnt+1

        if ep % plot_per_num_epoch == 0 or (ep + 1) == config.n_epochs:
        # if True:

            t2 = default_timer()
            loss_epoch/= ntrain
            loss_plot[ep] = loss_epoch
            
            plt.plot(np.arange(config.n_epochs)[:ep] , loss_plot[:ep], 'o-', linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.savefig(os.path.join(exp_path, 'Loss.jpg'))
            np.save(os.path.join(exp_path, 'Loss.npy'), loss_plot[:ep])
            plt.close()

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_exp_path)

            print('ep: {}/{} | time: {:.0f} | Loss: {:.6f} | GPU: {:.0f}'.format(ep, config.n_epochs, t2-t1,
                                                                                loss_epoch, config.gpu_num))
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('ep: {}/{} | time: {:.0f} | Loss: {:.6f} | gpu: {:.0f}'.format(ep, config.n_epochs, t2-t1,
                                                                                loss_epoch, config.gpu_num))
                file.write('\n')


            evaluator(ep = ep, subset = 'test', data_loader = test_loader,
                        model = model, exp_path = exp_path, operator = operator, operator_real = operator_real)



    
