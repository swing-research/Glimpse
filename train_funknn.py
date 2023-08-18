import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from funknn_model import FunkNN
from utils import *
from datasets import *
from results import evaluator_Bayesian, evaluator_sinogram
import config_funknn as config
import bayes as bnn

torch.manual_seed(0)
np.random.seed(0)

epochs_funknn = config.epochs_funknn
batch_size = config.batch_size
gpu_num = config.gpu_num
exp_desc = config.exp_desc
image_size = config.image_size
c = config.c
train_funknn = config.train_funknn
ood_analysis = config.ood_analysis

enable_cuda = True
device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

all_experiments = 'experiments/'
if os.path.exists(all_experiments) == False:
    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'funknn_' \
    + str(image_size) + '_' + str(config.c) + '_' + exp_desc


if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)



step_size = 50
gamma = 0.5
# myloss = F.mse_loss
myloss = F.l1_loss
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = config.kl_weight
num_batch_pixels = 3 # The number of iterations over each batch
batch_pixels = 512 # Number of pixels to optimize in each iteration
lam = 0 # TV norm coefficient

# Print the experiment setup:
print('Experiment setup:')
print('---> epochs_funknn: {}'.format(epochs_funknn))
print('---> batch_size: {}'.format(batch_size))
print('---> Learning rate: {}'.format(config.learning_rate))
print('---> experiment path: {}'.format(exp_path))
print('---> image size: {}'.format(image_size))

# Dataset:

# train_dataset = CT_CBP_loader(config.train_path)
# test_dataset = CT_CBP_loader(config.test_path)
# train_dataset = CT_CBP_generator(config.train_path, noise_snr = 200)
# test_dataset = CT_CBP_generator(config.test_path, noise_snr = 200)
train_dataset = CT_sinogram(config.train_path, noise_snr = config.train_noise_snr, self_supervised = False)
test_dataset = CT_sinogram(config.test_path, noise_snr = config.test_noise_snr, test_set= True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle = False)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)

n_ood = 0
if ood_analysis:
    # ood_dataset = CT_CBP_loader(config.ood_path)
    # ood_dataset = CT_CBP_generator(config.ood_path, noise_snr = 200)
    ood_dataset = CT_sinogram(config.ood_path, noise_snr = config.ood_noise_snr)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, num_workers=24)
    n_ood= len(ood_loader.dataset)

print('---> Number of training, test and ood samples: {}, {}, {}'.format(ntrain,n_test, n_ood))

# Loading model
plot_per_num_epoch = 1 if ntrain > 10000 else 30000//ntrain

model = FunkNN(c=c).to(device)
# model = torch.nn.DataParallel(model) # Using multiple GPUs
num_param_funknn = count_parameters(model)
print('---> Number of trainable parameters of funknn: {}'.format(num_param_funknn))

optimizer_funknn = Adam(model.parameters(), lr=config.learning_rate)
scheduler_funknn = torch.optim.lr_scheduler.StepLR(optimizer_funknn, step_size=step_size, gamma=gamma)

checkpoint_exp_path = os.path.join(exp_path, 'funknn.pt')
if os.path.exists(checkpoint_exp_path) and config.restore_funknn:
    checkpoint_funknn = torch.load(checkpoint_exp_path)
    model.load_state_dict(checkpoint_funknn['model_state_dict'])
    optimizer_funknn.load_state_dict(checkpoint_funknn['optimizer_state_dict'])
    print('funknn is restored...')

if train_funknn:
    print('Training...')

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = epochs_funknn + 1 # only plot in the last epoch
    
    loss_funknn_plot = np.zeros([epochs_funknn])
    for ep in range(epochs_funknn):
        model.train()
        t1 = default_timer()
        loss_funknn_epoch = 0

        for image, cbp in train_loader:
            
            batch_size = image.shape[0]
            image = image.to(device)
            cbp = cbp.to(device)
            
            for i in range(num_batch_pixels):

                coords = get_mgrid(image_size).reshape(-1, 2)
                coords = torch.unsqueeze(coords, dim = 0)
                coords = coords.expand(batch_size , -1, -1).to(device)
                
                optimizer_funknn.zero_grad()
                pixels = np.random.randint(low = 0, high = image_size**2, size = batch_pixels)
                batch_coords = coords[:,pixels]
                batch_image = image[:,pixels]

                out = model(batch_coords, cbp)
                mse_loss = myloss(out.reshape(batch_size, -1) , batch_image.reshape(batch_size, -1) )
                kl = kl_loss(model).to(device)
                total_loss = mse_loss  + kl_weight*kl

                total_loss.backward()
                optimizer_funknn.step()
                loss_funknn_epoch += total_loss.item()

        if ep % plot_per_num_epoch == 0 or (ep + 1) == epochs_funknn:

            scheduler_funknn.step()
            t2 = default_timer()
            loss_funknn_epoch/= ntrain
            loss_funknn_plot[ep] = loss_funknn_epoch
            
            plt.plot(np.arange(epochs_funknn)[:ep] , loss_funknn_plot[:ep], 'o-', linewidth=2)
            plt.title('FunkNN_loss')
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.savefig(os.path.join(exp_path, 'funknn_loss.jpg'))
            np.save(os.path.join(exp_path, 'funknn_loss.npy'), loss_funknn_plot[:ep])
            plt.close()

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_funknn.state_dict()}, checkpoint_exp_path)

            print('ep: {}/{} | time: {:.0f} | FunkNN_loss: {:.6f} | gpu: {:.0f}'.format(ep, epochs_funknn, t2-t1,
                                                                                   loss_funknn_epoch, config.gpu_num))
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('ep: {}/{} | time: {:.0f} | FunkNN_loss: {:.6f} | gpu: {:.0f}'.format(ep, epochs_funknn, t2-t1,
                                                                                   loss_funknn_epoch, config.gpu_num))
                file.write('\n')

            if config.Bayesian:
                evaluator_Bayesian(ep = ep, subset = 'test', data_loader = test_loader,
                          model = model, exp_path = exp_path)
            else:
                evaluator_sinogram(ep = ep, subset = 'test', data_loader = test_loader,
                          model = model, exp_path = exp_path)
                
            if ood_analysis:
                evaluator_sinogram(ep = ep, subset = 'ood', data_loader = ood_loader,
                    model = model, exp_path = exp_path)

print(model.alpha)
if config.Bayesian:
    evaluator_Bayesian(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)
else:
    evaluator_sinogram(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)

if ood_analysis:
    evaluator_sinogram(ep = -1, subset = 'ood', data_loader = ood_loader, model = model, exp_path = exp_path)




    
