from skimage.transform import resize
from skimage.transform import radon
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import imageio



def CT_sinogram(image_size = 128, n_angles = 30, 
                missing_cone = 'horizontal'):


    train_images_dir = '../datasets/CT/original_data/train'
    test_images_dir = '../datasets/CT/original_data/test'
    outlier_images_dir = 'datasets/CT_brain/test_samples/images'

    train_images_names = os.listdir(train_images_dir)
    test_images_names = os.listdir(test_images_dir)
    outlier_images_names = os.listdir(outlier_images_dir)

    n_train = len(train_images_names)
    n_test = len(test_images_names)
    n_outlier = len(outlier_images_names)
    print(n_train, n_test, n_outlier)

    data_folder = 'datasets/CT_CBP/'
    if os.path.exists(data_folder) == False:
        os.mkdir(data_folder)

    train_data_folder = data_folder +  f'train_{image_size}_{n_angles}_{missing_cone}_high_sinog/'
    if os.path.exists(train_data_folder) == False:
        os.mkdir(train_data_folder)

    test_data_folder = data_folder + f'test_{image_size}_{n_angles}_{missing_cone}_high_sinog/'
    if os.path.exists(test_data_folder) == False:
        os.mkdir(test_data_folder)

    outlier_data_folder = data_folder + f'outlier_{image_size}_{n_angles}_{missing_cone}_high_sinog/'
    if os.path.exists(outlier_data_folder) == False:
        os.mkdir(outlier_data_folder)

    np.random.seed(0)
    if missing_cone == 'horizontal':
        theta = np.linspace(30.0, 150.0, n_angles, endpoint=False)

    elif missing_cone == 'vertical':
        theta = np.linspace(-60.0, 60.0, n_angles, endpoint=False)

    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    # n_samples = n_test + n_train + n_outlier
    n_samples = n_test + n_outlier

    with tqdm(total=n_samples) as pbar:
        for i in range(n_samples):

            if i < n_outlier:
                image = imageio.imread(os.path.join(outlier_images_dir, outlier_images_names[i]))
                image = (image/255.0)

            elif i < n_test + n_outlier and i >= n_outlier :
                image = np.load(os.path.join(test_images_dir, test_images_names[i-n_outlier]))
                image_high = resize(image, (2*image_size,2*image_size))
            else:
                image = np.load(os.path.join(train_images_dir, train_images_names[i-n_outlier-n_test]))

            image = resize(image, (image_size,image_size))
            sinogram = radon(image, theta=theta, circle= False)
            if i == 0:
                plt.imsave(data_folder + 'image.png', image, cmap = 'gray')
                plt.imsave(data_folder + 'sinogram.png', sinogram, cmap = 'gray')
                print('First sample is saved.')
            
            if i < n_outlier:
                np.savez(outlier_data_folder + f'outlier_{i}.npz', image = image, sinogram = sinogram)
                pbar.set_description('outlier samples...')
                pbar.update(1)

            elif i < n_test + n_outlier and i >= n_outlier :
                np.savez(test_data_folder + f'test_{i-n_outlier}.npz', image = image,
                         sinogram = sinogram, image_high = image_high)
                pbar.set_description('test samples...')
                pbar.update(1)

            else:
                np.savez(train_data_folder + f'train_{i-n_outlier-n_test}.npz', image = image, sinogram = sinogram)
                pbar.set_description('train samples...')
                pbar.update(1)



if __name__ == '__main__':
    CT_sinogram(image_size = 128, missing_cone= 'complete', n_angles= 30)