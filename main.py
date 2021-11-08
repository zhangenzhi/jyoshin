import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from plotter import Plotter
from trainer import Trainer
import argparse
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot lossland around ture minima.')
    parser.add_argument(
        '--config', default='scripts/configs/2d_uniform_dnn.yaml')

    args = parser.parse_args()
    config = get_yml_content(args.config)

    trainer_args = config['Trainer']
    plotter_args = config['Plotter']

    trainer = Trainer(trainer_args)
    plotter = Plotter(plotter_args, trainer)

    plotter.run()
