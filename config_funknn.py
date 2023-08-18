epochs_funknn = 200 # number of epochs to train funknn network
batch_size = 64
gpu_num = 3 # GPU number
exp_desc = 'w_9_trainale_t' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
c = 30 # Number of channels of the dataset
train_funknn = True # Train or just reload to test
restore_funknn = False
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)
Bayesian = False
kl_weight = 0.01
num_training = -1
filter_init = 'ramp' # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
learnable_filter = False
missing_cone = 'complete'
w_size = 9
learning_rate = 1e-4
train_noise_snr = 200


# Evaluation arguements
max_scale = 2 # Maximum scale to generate in test time (2 or 4 or 8) (=<8 for celeba-hq and 2 for other datasets)
recursive = True # Recursive image reconstructions (Use just for factor training mode)
sample_number = 25 # Number of samples in evaluation
cmap = 'gray' # 'rgb' or for RGB images and other matplotlib colormaps for grayscales
derivatives_evaluation = False
num_posteriors = 25
test_noise_snr = 200
ood_noise_snr = 200


# Datasets paths:

ood_path = 'datasets/CT_CBP/outlier_128_30_complete_sinog/'
train_path = 'datasets/CT_CBP/train_128_30_complete_sinog/'
test_path = 'datasets/CT_CBP/test_128_30_complete_high_sinog/'

# ood_path = 'datasets/CT_CBP/outlier_128_45_vertical_sinog/'
# train_path = 'datasets/CT_CBP/train_128_45_vertical_sinog/'
# test_path = 'datasets/CT_CBP/test_128_45_vertical_sinog/'





