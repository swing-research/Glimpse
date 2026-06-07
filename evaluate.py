"""Evaluate a pretrained GLIMPSE checkpoint (no training).

Loads the checkpoint from the experiment directory (or ``--checkpoint``), then
reconstructs the test set (and the OOD set if enabled), writing figures and
PSNR/SSIM under ``<exp_path>/Results``::

    python evaluate.py --config configs/lodopab.yaml --checkpoint glimpse.pt
"""

import argparse
import os

import torch

from glimpse import Config
from glimpse.engine import build_model, evaluate, get_device, load_checkpoint
from glimpse.data import make_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained GLIMPSE model.")
    parser.add_argument('--config', type=str, default=None, help='YAML config file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (defaults to <exp_path>/glimpse.pt).')
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    device = get_device(config)
    true_angles, init_angles = config.resolve_angles()

    model = build_model(config, init_angles).to(device)
    checkpoint = args.checkpoint or os.path.join(config.exp_path, 'glimpse.pt')
    missing, unexpected = load_checkpoint(model, checkpoint, map_location=device)
    print(f'Loaded {checkpoint} (missing={missing}, unexpected={unexpected})')

    net = 'odl' if config.geometry == 'odl' else 'glimpse'
    operator = None
    if config.geometry == 'odl':
        from glimpse.operators import build_operator
        import numpy as np
        operator = build_operator(config, np.deg2rad(true_angles))

    test_set = make_dataset(config.test_path, config, true_angles, init_angles, network=net)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=8, shuffle=False)
    evaluate(config, 'test', test_loader, model, init_angles, epoch=-1, operator=operator)

    if config.ood_analysis:
        ood_set = make_dataset(config.ood_path, config, true_angles, init_angles, network=net)
        ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=config.batch_size, num_workers=8)
        evaluate(config, 'ood', ood_loader, model, init_angles, epoch=-1, operator=operator)


if __name__ == '__main__':
    main()
