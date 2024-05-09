import numpy as np
import ml_collections

n_epochs = 200 # number of epochs to train glimpse network
batch_size = 1
gpu_num = 2 # GPU number
exp_desc = 'base_learnf' # Add a small descriptor to the experiment
n1 = 800
n2 = 800
n3 = 300
train = True # Train or just reload to test
restore_model = True
filter_init = 'ramp' # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
learnable_filter = True # Learnable filter applied to sinogram
learning_rate = 1e-4
lsg = False  # Learnable sensore geomtery
cmap = 'gray' # 'rgb' or for RGB images and other matplotlib colormaps for grayscales
patch_shape = 'random'
learned_geo = True

train_path = '/local/Tomograms_cryoET/sup_data/general'
test_path = '/local/Tomograms_cryoET/sup_data/general'
ood_path = '../datasets/CT_brain/test_samples/images'

train_samples = ['model_model_1_res_6/', 
                'model_model_2_res_6/', 
                'model_model_3_res_6/', 
                'model_model_4_res_4/', 
                'model_model_5_res_5/', 
                'model_model_6_res_6/', 
                'model_model_7_res_6/',
                'model_model_8_res_6/',
                'model_model_9_res_4/',
                'model_model_10_res_4/',
                'model_model_11_res_5/',
                'model_model_12_res_6/',
                'model_model_13_res_4/',
                'model_model_14_res_4/',
                'model_model_15_res_5/',
                'model_model_16_res_5/',
                'model_model_17_res_6/',
                'model_model_18_res_4/',
                'model_model_19_res_6/',
                'model_model_20_res_5/',
                'model_model_21_res_6/',
                'model_model_22_res_4/']

test_samples = ['model_model_27_res_5/',
                'model_model_28_res_5/',
                'model_model_29_res_6/',
                'model_model_30_res_6/']


data = ml_collections.ConfigDict()
# Note the angles max and min are used to create the ramp filter using odl
data.angle_max = np.pi/3
data.angle_min =  -np.pi/3
data.n_projections = 41
data.simulate_noise = True
data.noise_level = 0 #[-10,5] # db
data.pix = 8
data.fixed_angles = True
data.defocus_list = [-3, -4, -5, -6]
data.dose_list =  [100,200,300,400,500,50,60]
data.angles = np.linspace(data.angle_min, data.angle_max, data.n_projections)