from skimage.transform import resize
from skimage.transform import radon, iradon
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import imageio
import torch
import torch.nn.functional as F



def CT_sinogram(image_size = 128, n_angles = 30, 
                missing_cone = 'complete', noise_snr = 30):
    

    gpu_num = 0
    device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else 'cpu')

    train_images_dir = '../../datasets/CT/original_data/train'
    test_images_dir = '../../datasets/CT/original_data/test'
    outlier_images_dir = '../datasets/CT_brain/test_samples/images'

    train_images_names = os.listdir(train_images_dir)
    test_images_names = os.listdir(test_images_dir)
    outlier_images_names = os.listdir(outlier_images_dir)

    n_train = len(train_images_names)
    n_test = len(test_images_names)
    n_outlier = len(outlier_images_names)
    print(n_train, n_test, n_outlier)

    data_folder = f'datasets/{image_size}_{n_angles}_{missing_cone}_{noise_snr}/'
    if os.path.exists(data_folder) == False:
        os.mkdir(data_folder)

    train_data_folder = data_folder +  f'train/'
    if os.path.exists(train_data_folder) == False:
        os.mkdir(train_data_folder)

    test_data_folder = data_folder + f'test/'
    if os.path.exists(test_data_folder) == False:
        os.mkdir(test_data_folder)

    outlier_data_folder = data_folder + f'outlier/'
    if os.path.exists(outlier_data_folder) == False:
        os.mkdir(outlier_data_folder)

    np.random.seed(0)
    if missing_cone == 'horizontal':
        theta = np.linspace(30.0, 150.0, n_angles, endpoint=False)

    elif missing_cone == 'vertical':
        theta = np.linspace(-60.0, 60.0, n_angles, endpoint=False)

    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    n_samples = n_test + n_train + n_outlier
    # n_samples = n_test + n_outlier

    with tqdm(total=n_samples) as pbar:
        for i in range(n_samples):

            if i < n_outlier:
                image = imageio.imread(os.path.join(outlier_images_dir, outlier_images_names[i]))
                image = (image/255.0)

            elif i < n_test + n_outlier and i >= n_outlier :
                image = np.load(os.path.join(test_images_dir, test_images_names[i-n_outlier]))
            else:
                image = np.load(os.path.join(train_images_dir, train_images_names[i-n_outlier-n_test]))

            # image = resize(image, (image_size,image_size))
            image = torch.tensor(image, dtype = torch.float32)[None,None].to(device)
            image = F.interpolate(image, size = image_size,
                                    mode = 'bilinear',
                                    antialias= True,
                                    align_corners= True)[0,0].cpu().detach().numpy()
            
            sinogram = radon(image, theta=theta, circle= False)
            noise_sigma = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
            noise = np.random.normal(loc = 0,
                                     scale = noise_sigma,
                                     size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
            sinogram += noise

            fbp = iradon(sinogram, theta=theta, circle= False)


            if i == 0:
                plt.imsave(data_folder + 'image.png', image, cmap = 'gray')
                plt.imsave(data_folder + 'sinogram.png', sinogram, cmap = 'gray')
                plt.imsave(data_folder + 'fbp.png', fbp, cmap = 'gray')
                print('First sample is saved.')
            
            if i < n_outlier:
                np.savez(outlier_data_folder + f'outlier_{i}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('outlier samples...')
                pbar.update(1)

            elif i < n_test + n_outlier and i >= n_outlier :
                np.savez(test_data_folder + f'test_{i-n_outlier}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('test samples...')
                pbar.update(1)

            else:
                np.savez(train_data_folder + f'train_{i-n_outlier-n_test}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('train samples...')
                pbar.update(1)



if __name__ == '__main__':
    CT_sinogram(image_size = 128,
                missing_cone= 'complete',
                n_angles= 30,
                noise_snr= 20)