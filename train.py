"""Train (and evaluate) GLIMPSE.

Configuration comes from a YAML file plus optional command-line overrides::

    python train.py --config configs/lodopab.yaml
    python train.py --config configs/lodopab.yaml --n-angles 60 --epochs 5000

With no ``--config`` the built-in defaults in ``glimpse.config.Config`` are used.
"""

from glimpse import Config
from glimpse.engine import train


def main():
    config = Config.from_args()
    train(config)


if __name__ == '__main__':
    main()
