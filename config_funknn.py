import numpy as np

n_epochs = 200 # number of epochs to train funknn network
batch_size = 64
gpu_num = 1 # GPU number
exp_desc = 'base_no_interference' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
n_angles = 30 # Number of channels of the dataset
train = True # Train or just reload to test
restore_model = True
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)
num_training = -1
filter_init = 'hamming' # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
learnable_filter = False
missing_cone = 'complete'
w_size = 9
learning_rate = 1e-4
train_noise_snr = 30
activation = 'relu'
memory_analysis = False
multiscale = False
scale = 8
uncalibrated = False
lsg = True

if missing_cone == 'horizontal':
    theta = np.linspace(30.0, 150.0, n_angles, endpoint=False)

elif missing_cone == 'vertical':
    theta = np.linspace(-60.0, 60.0, n_angles, endpoint=False)

else:
    if uncalibrated:
        # theta = np.linspace(3.0, 183.0, n_angles, endpoint=False)
        np.random.seed(2)
        theta = np.random.rand(n_angles) * 180.0
        theta = np.sort(theta)
    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    


# Evaluation arguements
sample_number = 25 # Number of samples in evaluation
cmap = 'gray' # 'rgb' or for RGB images and other matplotlib colormaps for grayscales
test_noise_snr = 30
ood_noise_snr = 30


# Datasets paths:

train_path = 'datasets/128_30_complete_30_right/train'
test_path = 'datasets/128_30_complete_30_right/test'
ood_path = 'datasets/128_30_complete_30_right/outlier'

# train_path = 'datasets/512_30_complete_40_right/train'
# test_path = 'datasets/512_30_complete_40_right/test'
# ood_path = 'datasets/512_30_complete_40_right/outlier'

# train_path = '../../datasets/CT/original_data/train'
# test_path = '../../datasets/CT/original_data/test'
# ood_path = '../datasets/CT_brain/test_samples/images'






