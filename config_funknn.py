import numpy as np

n_epochs = 200 # number of epochs to train funknn network
batch_size = 64
gpu_num = 0 # GPU number
exp_desc = '0.5_receptive_field' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
n_angles = 30 # Number of channels of the dataset
train = True # Train or just reload to test
restore_model = True
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)
num_training = -1
filter_init = 'ramp' # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
learnable_filter = True
missing_cone = 'complete'
w_size = 9
learning_rate = 1e-4
train_noise_snr = 30
activation = 'relu'
if missing_cone == 'horizontal':
    theta = np.linspace(30.0, 150.0, n_angles, endpoint=False)

elif missing_cone == 'vertical':
    theta = np.linspace(-60.0, 60.0, n_angles, endpoint=False)

else:
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)


# Evaluation arguements
sample_number = 25 # Number of samples in evaluation
cmap = 'gray' # 'rgb' or for RGB images and other matplotlib colormaps for grayscales
test_noise_snr = 30
ood_noise_snr = 30


# Datasets paths:


train_path = '../../datasets/CT/original_data/train'
test_path = '../../datasets/CT/original_data/test'
ood_path = '../datasets/CT_brain/test_samples/images'

# ood_path = '../datasets/CT_CBP/outlier_128_30_complete_sinog/'
# train_path = '../datasets/CT_CBP/train_128_30_complete_sinog/'
# test_path = '../datasets/CT_CBP/test_128_30_complete_high_sinog/'

# ood_path = '../datasets/CT_CBP/outlier_128_45_vertical_sinog/'
# train_path = '../datasets/CT_CBP/train_128_45_vertical_sinog/'
# test_path = '../datasets/CT_CBP/test_128_45_vertical_sinog/'






