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

torch.manual_seed(0)
np.random.seed(0)


enable_cuda = True
device = torch.device('cuda:' + str(config.gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

# experiment path
exp_path = 'experiments/' \
    + str(config.image_size) + '_' + str(config.n_angles) + '_' + config.exp_desc
os.makedirs(exp_path, exist_ok=True)


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
train_dataset = CT_dataset(config.train_path)
test_dataset = CT_dataset(config.test_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=24, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=24, shuffle = False)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)

n_ood = 0
if config.ood_analysis:
    ood_dataset = CT_dataset(config.ood_path)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=config.batch_size, num_workers=24, shuffle = False)
    n_ood= len(ood_loader.dataset)

print('---> Number of training, test and ood samples: {}, {}, {}'.format(ntrain,n_test, n_ood))

# Loading model
plot_per_num_epoch = 1 if ntrain > 10000 else 30000//ntrain

model = glimpse(image_size = config.image_size, w_size = config.w_size,
                theta_init = config.theta_init, lsg = config.lsg,
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


if config.train:
    print('Training...')

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = config.n_epochs + 1 # only plot in the last epoch
    
    loss_plot = np.zeros([config.n_epochs])
    for ep in range(config.n_epochs):
        model.train()
        t1 = default_timer()
        loss_epoch = 0

        for image, sinogram in train_loader:
            
            batch_size = image.shape[0]
            image = image.to(device)
            sinogram = sinogram.to(device)

            model.train()
            
            for i in range(num_batch_pixels):

                coords = get_mgrid(config.image_size).reshape(-1, 2)
                coords = torch.unsqueeze(coords, dim = 0)
                coords = coords.expand(batch_size , -1, -1).to(device)
                
                optimizer.zero_grad()
                pixels = np.random.randint(low = 0, high = config.image_size**2, size = batch_pixels)
                batch_coords = coords[:,pixels]
                batch_image = image[:,pixels]

                out = model(batch_coords, sinogram)
                mse_loss = myloss(out.reshape(batch_size, -1) , batch_image.reshape(batch_size, -1) )
                total_loss = mse_loss

                total_loss.backward()
                optimizer.step()
                loss_epoch += total_loss.item()

        if ep % plot_per_num_epoch == 0 or (ep + 1) == config.n_epochs:

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
                        model = model, exp_path = exp_path)
            if config.ood_analysis:
                evaluator(ep = ep, subset = 'ood', data_loader = ood_loader,
                    model = model, exp_path = exp_path)

evaluator(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)
if config.ood_analysis:
    evaluator(ep = -1, subset = 'ood', data_loader = ood_loader, model = model, exp_path = exp_path)




    
