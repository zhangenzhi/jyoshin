from trainer import Trainer
from plotter import Plotter
import time
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 100, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh'], 'fuse_models': 1000},
                    }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    # trainer.uniform_self_evaluate()

    # plotter_args = {'num_evaluate': 10,
    #                 'step': 1/10,
    #                 'fuse_models': trainer_args['model']['fuse_models'],
    #                 }
    # plotter = Plotter(plotter_args, trainer.model)

    # # 1d-loss
    # plotter.plot_1d_loss(trainer=trainer)

    plotter_args = {'num_evaluate': [10, 1],
                    'step': [1e-1, 1e-3],
                    'fuse_models': trainer_args['model']['fuse_models'],
                    }
    plotter = Plotter(plotter_args, trainer.model)

    # 2d-loss
    plotter.plot_2d_loss(trainer=trainer)
