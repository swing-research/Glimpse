"""U-Net baseline for sparse-view CT reconstruction.

A post-processing baseline to compare against GLIMPSE: it learns the mapping
FBP-image -> clean-image with a standard U-Net (loaded from torch.hub). Unlike
GLIMPSE it operates on full images and does not generalize across resolutions or
geometries, but it is a strong learned denoiser of FBP artefacts.

Run directly with command-line arguments::

    python baselines/unet.py --train-path datasets/train --test-path datasets/test \
        --image-size 128 --n-angles 30 --epochs 200

This script is self-contained and does NOT use glimpse/config.py; it only reuses
the dataset loaders and metrics from the ``glimpse`` package.
"""

import argparse
import os
import sys
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Allow running as a plain script from the repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glimpse.data import make_dataset  # noqa: E402
from glimpse.config import Config  # noqa: E402
from glimpse.metrics import PSNR, SSIM, count_parameters  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="U-Net FBP-postprocessing baseline.")
    p.add_argument('--train-path', default='datasets/train')
    p.add_argument('--test-path', default='datasets/test')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--n-angles', type=int, default=30)
    p.add_argument('--noise-snr', type=float, default=30)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--exp-path', default='experiments/unet')
    p.add_argument('--restore', action='store_true', help='Reload model.pt before training.')
    p.add_argument('--no-train', action='store_true', help='Only evaluate.')
    return p.parse_args()


def save_grid(path, images, n_grid, image_size, cmap='gray'):
    grid = images[:n_grid * n_grid].reshape(
        n_grid, n_grid, image_size, image_size, 1
    ).swapaxes(1, 2).reshape(n_grid * image_size, -1, 1)
    plt.imsave(path, grid[:, :, 0], cmap=cmap)


def evaluate(model, loader, device, results_path, image_size, epoch=-1):
    model.eval()
    x_test, y_test = next(iter(loader))
    x_test, y_test = x_test.to(device), y_test.to(device)
    t1 = default_timer()
    with torch.no_grad():
        recon = model(y_test)
    print(f'Inference time: {default_timer() - t1:.3f}s')

    x = x_test.cpu().numpy()[:, 0]
    y = y_test.cpu().numpy()[:, 0]
    r = recon.cpu().numpy()[:, 0]
    os.makedirs(results_path, exist_ok=True)
    n_grid = 1
    save_grid(os.path.join(results_path, f'{epoch}_gt.png'), x, n_grid, image_size)
    save_grid(os.path.join(results_path, f'{epoch}_fbp.png'), y, n_grid, image_size)
    save_grid(os.path.join(results_path, f'{epoch}_unet.png'), r, n_grid, image_size)
    print('U-Net PSNR: {:.1f} | FBP PSNR: {:.1f} | U-Net SSIM: {:.2f} | FBP SSIM: {:.2f}'.format(
        PSNR(x, r), PSNR(x, y), SSIM(x, r), SSIM(x, y)))


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.exp_path, exist_ok=True)
    results_path = os.path.join(args.exp_path, 'results')

    # Angles: U-Net trains/evaluates on the calibrated geometry.
    angles = np.linspace(0.0, 180.0, args.n_angles, endpoint=False)
    cfg = Config(image_size=args.image_size, n_angles=args.n_angles, noise_snr=args.noise_snr)

    # The U-Net consumes (clean_image, fbp_image) pairs -> network='unet'.
    train_set = make_dataset(args.train_path, cfg, angles, angles, network='unet')
    test_set = make_dataset(args.test_path, cfg, angles, angles, network='unet')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=24, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, num_workers=24)

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=1, init_features=32,
                           pretrained=False).to(device)
    print(f'---> U-Net trainable parameters: {count_parameters(model)}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(args.exp_path, 'model.pt')
    if os.path.exists(checkpoint_path) and args.restore:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print('U-Net restored...')

    if not args.no_train:
        n_train = len(train_loader.dataset)
        plot_per = 1 if n_train > 10000 else max(1, 30000 // n_train)
        for ep in range(args.epochs):
            model.train()
            start = default_timer()
            loss_epoch = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(model(y).reshape(x.shape[0], -1), x.reshape(x.shape[0], -1))
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            if ep % plot_per == 0 or (ep + 1) == args.epochs:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                print('epoch {}/{} | time {:.0f}s | loss {:.6f}'.format(
                    ep, args.epochs, default_timer() - start, loss_epoch / n_train))
                evaluate(model, test_loader, device, results_path, args.image_size, epoch=ep)

    evaluate(model, test_loader, device, results_path, args.image_size, epoch=-1)


if __name__ == '__main__':
    main()
