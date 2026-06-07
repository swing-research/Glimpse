"""Training and evaluation loops for GLIMPSE.

``build_model`` / ``load_checkpoint`` construct the network and restore weights;
``evaluate`` writes reconstruction figures and PSNR/SSIM; ``train`` runs the full
per-pixel training loop. All take an explicit :class:`~glimpse.config.Config`
instead of reading module-level globals.
"""

import os
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .config import Config
from .data import make_dataset
from .metrics import PSNR, SSIM, count_parameters
from .model import GlimpseModel
from .reconstruct import fbp_batch, make_coordinate_grid, reconstruct_image


def get_device(config: Config) -> torch.device:
    return torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')


def build_model(config: Config, init_angles_deg) -> GlimpseModel:
    """Instantiate a GlimpseModel from a Config and its initial angles."""
    return GlimpseModel(
        image_size=config.image_size, patch_size=config.patch_size,
        # list (not ndarray) so the HF Hub mixin can serialise it to config.json
        init_angles_deg=[float(a) for a in init_angles_deg],
        learn_geometry=config.learn_geometry,
        learnable_filter=config.learnable_filter, filter_name=config.filter_name,
        head_type=config.head_type, geometry=config.geometry, circle=config.circle,
        patch_shape=config.patch_shape, learn_patch=config.learn_patch)


def load_checkpoint(model, path, optimizer=None, map_location=None, strict=False):
    """Load a ``glimpse.pt``-style checkpoint into ``model`` (and optimizer).

    With ``sampler='affine'`` (the default, matching the published checkpoint)
    the keys line up exactly. ``strict=False`` is kept as a convenience so a
    ``sampler='patch'`` model can still load the checkpoint (its ``patch`` /
    ``patch_scale`` then fall back to their initialisation). Returns the
    ``(missing, unexpected)`` key lists.
    """
    checkpoint = torch.load(path, map_location=map_location)
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    missing = list(getattr(result, 'missing_keys', []))
    unexpected = list(getattr(result, 'unexpected_keys', []))
    return missing, unexpected


def _save_grid(path, images, n_grid, image_size, cmap):
    """Tile the first ``n_grid**2`` images into a square grid PNG."""
    grid = images[:n_grid * n_grid].reshape(
        n_grid, n_grid, image_size, image_size, 1
    ).swapaxes(1, 2).reshape(n_grid * image_size, -1, 1)
    plt.imsave(path, grid[:, :, 0], cmap=cmap)


def evaluate(config: Config, subset, data_loader, model, fbp_angles_deg, epoch=-1, operator=None):
    """Reconstruct one batch, write GT/FBP/GLIMPSE/error PNGs, log PSNR & SSIM.

    For ``geometry='odl'`` an ``operator`` (:class:`~glimpse.operators.
    ParallelBeam2DOperator`) must be supplied; the loader then yields raw images
    and the sinogram + FBP baseline are produced by the operator.
    """
    exp_path = config.exp_path
    results_folder = os.path.join(exp_path, 'Results')
    os.makedirs(results_folder, exist_ok=True)
    device = next(model.parameters()).device
    image_size = config.image_size

    if subset == 'ood':
        n_write, n_grid = 16, 4
    else:
        n_grid = int(np.sqrt(config.sample_number if config.sample_number < 26 else 25))
        n_write = int(n_grid ** 2)

    print(f'Evaluation on {subset} set:')
    with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
        f.write(f'Evaluation on {subset} set:\n')

    images, second = next(iter(data_loader))
    images = images.to(device).reshape(-1, image_size, image_size, 1)
    images_np = images.detach().cpu().numpy()[:, :, :, 0]

    # Sinogram + FBP baseline (geometry-dependent).
    if config.geometry == 'odl':
        sinogram = operator.project(second.to(device))
        fbp = operator.fbp(sinogram).detach().cpu().numpy()
    else:
        sinogram = second.to(device)
        fbp = fbp_batch(sinogram.detach().cpu().numpy(), theta=fbp_angles_deg)

    _save_grid(os.path.join(results_folder, f'{epoch}_{subset}_gt.png'),
               images_np, n_grid, image_size, config.cmap)
    _save_grid(os.path.join(results_folder, f'{epoch}_{subset}_fbp.png'),
               fbp, n_grid, image_size, config.cmap)

    # GLIMPSE reconstruction.
    model.eval()
    coords = make_coordinate_grid(image_size).unsqueeze(0).expand(images.shape[0], -1, -1).to(device)
    t1 = default_timer()
    recon = reconstruct_image(sinogram, coords, 1, model, chunk_size=1024)
    print(f'Elapsed inference time: {default_timer() - t1:.3f}s')
    if device.type == 'cuda':
        print(f"Maximum GPU memory used: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")
    recon = np.reshape(recon, [-1, image_size, image_size, 1])[:, :, :, 0]
    _save_grid(os.path.join(results_folder, f'{epoch}_{subset}_glimpse.png'),
               recon, n_grid, image_size, config.cmap)

    # Error map.
    error = np.abs(images_np - recon)
    _save_grid(os.path.join(results_folder, f'{epoch}_{subset}_error.png'),
               error, n_grid, image_size, 'seismic')

    psnr_recon, psnr_fbp = PSNR(images_np, recon), PSNR(images_np, fbp)
    ssim_recon, ssim_fbp = SSIM(images_np, recon), SSIM(images_np, fbp)
    np.savez(os.path.join(exp_path, f'glimpse_{subset}.npz'),
             images=images_np, fbp=fbp, glimpse=recon)

    line = 'PSNR fbp: {:.1f} | PSNR glimpse: {:.1f} | SSIM fbp: {:.2f} | SSIM glimpse: {:.2f}'.format(
        psnr_fbp, psnr_recon, ssim_fbp, ssim_recon)
    print(line)
    with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
        f.write(line + '\n')
        if subset == 'ood':
            f.write('\n')
    return {'psnr_fbp': psnr_fbp, 'psnr_glimpse': psnr_recon,
            'ssim_fbp': ssim_fbp, 'ssim_glimpse': ssim_recon}


def train(config: Config):
    """Full training/evaluation entry point driven by ``config``."""
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = get_device(config)
    exp_path = config.exp_path
    os.makedirs(exp_path, exist_ok=True)
    true_angles, init_angles = config.resolve_angles()

    print('Experiment setup:')
    print(f'---> epochs: {config.epochs} | batch_size: {config.batch_size} | lr: {config.lr}')
    print(f'---> image size: {config.image_size} | n_angles: {config.n_angles} | geometry: {config.geometry}')
    print(f'---> experiment path: {exp_path}')

    # For geometry='odl' the loaders yield raw images and the ODL operator
    # generates sinograms on the GPU each step; for skimage they yield sinograms.
    net = 'odl' if config.geometry == 'odl' else 'glimpse'
    operator = None
    if config.geometry == 'odl':
        from .operators import build_operator
        operator = build_operator(config, np.deg2rad(true_angles))

    train_set = make_dataset(config.train_path, config, true_angles, init_angles, network=net)
    test_set = make_dataset(config.test_path, config, true_angles, init_angles, network=net)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, num_workers=config.num_workers, shuffle=False)
    n_train, n_test = len(train_loader.dataset), len(test_loader.dataset)

    ood_loader = None
    n_ood = 0
    if config.ood_analysis:
        ood_set = make_dataset(config.ood_path, config, true_angles, init_angles, network=net)
        ood_loader = torch.utils.data.DataLoader(
            ood_set, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        n_ood = len(ood_loader.dataset)
    print(f'---> train / test / ood samples: {n_train}, {n_test}, {n_ood}')

    model = build_model(config, init_angles).to(device)
    print(f'---> trainable parameters: {count_parameters(model)}')
    optimizer = Adam(model.parameters(), lr=config.lr)

    checkpoint_path = os.path.join(exp_path, 'glimpse.pt')
    if os.path.exists(checkpoint_path) and config.restore_model:
        load_checkpoint(model, checkpoint_path, optimizer=optimizer, map_location=device)
        print('glimpse checkpoint restored...')

    evaluate(config, 'test', test_loader, model, init_angles, epoch=-1, operator=operator)
    if ood_loader is not None:
        evaluate(config, 'ood', ood_loader, model, init_angles, epoch=-1, operator=operator)

    if not config.train:
        return model

    print('Training...')
    plot_per_num_epoch = 1 if n_train > 10000 else max(1, 30000 // n_train)
    loss_plot = np.zeros([config.epochs])
    coords_full = make_coordinate_grid(config.image_size).reshape(-1, 2)

    for ep in range(config.epochs):
        model.train()
        t1 = default_timer()
        loss_epoch = 0.0
        for image, second in train_loader:
            batch_size = image.shape[0]
            image = image.to(device)
            if config.geometry == 'odl':
                sinogram = operator.project(second.to(device))
            else:
                sinogram = second.to(device)
            coords = coords_full.unsqueeze(0).expand(batch_size, -1, -1).to(device)

            for _ in range(config.steps_per_batch):
                optimizer.zero_grad()
                pixels = np.random.randint(0, config.image_size ** 2, size=config.pixels_per_step)
                out = model(coords[:, pixels], sinogram)
                loss = F.mse_loss(out.reshape(batch_size, -1), image[:, pixels].reshape(batch_size, -1))
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

        if ep % plot_per_num_epoch == 0 or (ep + 1) == config.epochs:
            loss_epoch /= n_train
            loss_plot[ep] = loss_epoch
            plt.plot(np.arange(config.epochs)[:ep], loss_plot[:ep], 'o-', linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.savefig(os.path.join(exp_path, 'Loss.jpg'))
            np.save(os.path.join(exp_path, 'Loss.npy'), loss_plot[:ep])
            plt.close()
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print('ep: {}/{} | time: {:.0f}s | Loss: {:.6f}'.format(
                ep, config.epochs, default_timer() - t1, loss_epoch))
            with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
                f.write('ep: {}/{} | time: {:.0f} | Loss: {:.6f}\n'.format(
                    ep, config.epochs, default_timer() - t1, loss_epoch))

            evaluate(config, 'test', test_loader, model, init_angles, epoch=ep, operator=operator)
            if ood_loader is not None:
                evaluate(config, 'ood', ood_loader, model, init_angles, epoch=ep, operator=operator)

    return model
