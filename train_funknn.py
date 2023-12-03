import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from funknn_model import Deep_local
from utils import *
from data_loader import *
from results import evaluator_sinogram
import config_funknn as config

torch.manual_seed(0)
np.random.seed(0)


enable_cuda = True
device = torch.device('cuda:' + str(config.gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

all_experiments = 'experiments/'
if os.path.exists(all_experiments) == False:
    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments \
    + str(config.image_size) + '_' + str(config.n_angles) + '_' + config.exp_desc

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


step_size = 50
gamma = 0.5
myloss = F.mse_loss
# myloss = F.l1_loss
num_batch_pixels = 3 # The number of iterations over each batch
batch_pixels = 512 # Number of pixels to optimize in each iteration

# Print the experiment setup:
print('Experiment setup:')
print('---> num epochs: {}'.format(config.n_epochs))
print('---> batch_size: {}'.format(config.batch_size))
print('---> Learning rate: {}'.format(config.learning_rate))
print('---> experiment path: {}'.format(exp_path))
print('---> image size: {}'.format(config.image_size))

# Dataset:
train_dataset = CT_dataset(config.train_path, train = True)
test_dataset = CT_dataset(config.test_path, train = False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=24, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=24, shuffle = False)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)

n_ood = 0
if config.ood_analysis:
    ood_dataset = CT_dataset(config.ood_path, train = False)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=config.batch_size, num_workers=24, shuffle = False)
    n_ood= len(ood_loader.dataset)

print('---> Number of training, test and ood samples: {}, {}, {}'.format(ntrain,n_test, n_ood))

# Loading model
plot_per_num_epoch = 1 if ntrain > 10000 else 30000//ntrain

model = Deep_local().to(device)
# model = torch.nn.DataParallel(model) # Using multiple GPUs
num_param_funknn = count_parameters(model)
print('---> Number of trainable parameters of funknn: {}'.format(num_param_funknn))

optimizer_funknn = Adam(model.parameters(), lr=config.learning_rate)
scheduler_funknn = torch.optim.lr_scheduler.StepLR(optimizer_funknn, step_size=step_size, gamma=gamma)

theta_rad = model.theta_rad
theta_deg = np.rad2deg(theta_rad.detach().cpu().numpy())
np.save(os.path.join(exp_path, 'init_sensors_locs.npy'), theta_deg)
print(theta_deg)

filter = model.fourier_filter
filter = filter.detach().cpu().numpy()
np.save(os.path.join(exp_path, 'init_filter.npy'), filter)

checkpoint_exp_path = os.path.join(exp_path, 'funknn.pt')
if os.path.exists(checkpoint_exp_path) and config.restore_model:
    checkpoint_funknn = torch.load(checkpoint_exp_path)
    model.load_state_dict(checkpoint_funknn['model_state_dict'])
    optimizer_funknn.load_state_dict(checkpoint_funknn['optimizer_state_dict'])
    print('funknn is restored...')


theta_rad = model.theta_rad
theta_deg = np.rad2deg(theta_rad.detach().cpu().numpy())
print(theta_deg)
np.save(os.path.join(exp_path, 'sensors_locs.npy'), theta_deg)

filter = model.fourier_filter
filter = filter.detach().cpu().numpy()
np.save(os.path.join(exp_path, 'learned_filter.npy'), filter)


if config.train:
    print('Training...')

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = config.n_epochs + 1 # only plot in the last epoch
    
    loss_funknn_plot = np.zeros([config.n_epochs])
    for ep in range(config.n_epochs):
        model.train()
        t1 = default_timer()
        loss_funknn_epoch = 0

        for image, sinogram in train_loader:
            
            batch_size = image.shape[0]
            image = image.to(device)
            sinogram = sinogram.to(device)
            # fbp = fbp.to(device)

            model.train()
            
            for i in range(num_batch_pixels):

                coords = get_mgrid(config.image_size).reshape(-1, 2)
                coords = torch.unsqueeze(coords, dim = 0)
                coords = coords.expand(batch_size , -1, -1).to(device)
                
                optimizer_funknn.zero_grad()
                pixels = np.random.randint(low = 0, high = config.image_size**2, size = batch_pixels)
                batch_coords = coords[:,pixels]
                batch_image = image[:,pixels]

                # shift = np.random.randint(low = 0, high = 4)
                # mirror = True if np.random.rand(1) > 0.5 else False
                # shift = 0
                out = model(batch_coords, sinogram)
                mse_loss = myloss(out.reshape(batch_size, -1) , batch_image.reshape(batch_size, -1) )
                total_loss = mse_loss

                total_loss.backward()
                optimizer_funknn.step()
                loss_funknn_epoch += total_loss.item()

        if ep % plot_per_num_epoch == 0 or (ep + 1) == config.n_epochs:

            # scheduler_funknn.step()
            t2 = default_timer()
            loss_funknn_epoch/= ntrain
            loss_funknn_plot[ep] = loss_funknn_epoch
            
            plt.plot(np.arange(config.n_epochs)[:ep] , loss_funknn_plot[:ep], 'o-', linewidth=2)
            plt.title('FunkNN_loss')
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.savefig(os.path.join(exp_path, 'funknn_loss.jpg'))
            np.save(os.path.join(exp_path, 'funknn_loss.npy'), loss_funknn_plot[:ep])
            plt.close()

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_funknn.state_dict()}, checkpoint_exp_path)

            print('ep: {}/{} | time: {:.0f} | FunkNN_loss: {:.6f} | gpu: {:.0f}'.format(ep, config.n_epochs, t2-t1,
                                                                                   loss_funknn_epoch, config.gpu_num))
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('ep: {}/{} | time: {:.0f} | FunkNN_loss: {:.6f} | gpu: {:.0f}'.format(ep, config.n_epochs, t2-t1,
                                                                                   loss_funknn_epoch, config.gpu_num))
                file.write('\n')


            evaluator_sinogram(ep = ep, subset = 'test', data_loader = test_loader,
                        model = model, exp_path = exp_path)
            if config.ood_analysis:
                evaluator_sinogram(ep = ep, subset = 'ood', data_loader = ood_loader,
                    model = model, exp_path = exp_path)

evaluator_sinogram(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)
if config.ood_analysis:
    evaluator_sinogram(ep = -1, subset = 'ood', data_loader = ood_loader, model = model, exp_path = exp_path)




    
