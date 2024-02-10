import numpy as np

n_epochs = 200 # number of epochs to train glimpse network
batch_size = 64
gpu_num = 0 # GPU number
exp_desc = 'default' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
n_angles = 30 # Number of channels of the dataset
noise_snr = 30
train = True # Train or just reload to test
restore_model = True
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)
filter_init = 'ramp' # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
learnable_filter = True # Learnable filter applied to sinogram
w_size = 9
learning_rate = 1e-4
uncalibrated_type = 'No' # No: calibrated, random: randomly shifted projections
# fixed: fixed shift in projection angles, blind: no information from projection angles
lsg = True  # Learnable sensore geomtery
sample_number = 25 # Number of samples in used in visualization
cmap = 'gray' # 'rgb' or for RGB images and other matplotlib colormaps for grayscales
theta_actual = np.linspace(0.0, 180.0, n_angles, endpoint=False)


np.random.seed(2)
if uncalibrated_type == 'No':
    theta_init = theta_actual

elif uncalibrated_type == 'random':
    # uncalibrated random
    shifts = np.random.randn(n_angles) * 2.0
    theta_init = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    theta_init = theta_init + shifts

elif uncalibrated_type == 'fixed':
    # uncalibrated shifts:
    theta_init = np.linspace(3.0, 183.0, n_angles, endpoint=False)

elif uncalibrated_type == 'blind':
    # blind
    theta_init = np.random.rand(n_angles) * 180.0
    theta_init = np.sort(theta_init)


# Datasets paths:

# train_path = 'datasets/128_30_complete_40/train'
# test_path = 'datasets/128_30_complete_40/test'
# ood_path = 'datasets/128_30_complete_40/outlier'

# train_path = 'datasets/128_30_complete_30_right/train'
# test_path = 'datasets/128_30_complete_30_right/test'
# ood_path = 'datasets/128_30_complete_30_right/outlier'

# train_path = 'datasets/512_30_complete_40_right/train'
# test_path = 'datasets/512_30_complete_40_right/test'
# ood_path = 'datasets/512_30_complete_40_right/outlier'

train_path = '../../datasets/CT/original_data/train'
test_path = '../../datasets/CT/original_data/test'
ood_path = '../datasets/CT_brain/test_samples/images'

