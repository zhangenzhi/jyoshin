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
                              'activations': ['tanh', 'tanh', 'tanh'], 'fuse_models': 500},
                    }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    # trainer.uniform_self_evaluate()

    plotter_args = {'num_evaluate': 20000,
                    'step': 1/1000000,
                    'fuse_models': trainer_args['model']['fuse_models'],
                    }
    plotter = Plotter(plotter_args, trainer.model)

    # set init state
    fused_direction, normalized_direction = plotter.create_random_direction(
        norm='layer')
    plotter.set_weights(init_state=True, init_directions=normalized_direction)
    trainer.uniform_self_evaluate()
    # import pdb
    # pdb.set_trace()

    # plot num_evaluate * fuse_models points in lossland
    start_time = time.time()
    for i in range(plotter.num_evaluate):
        plotter.set_weights(directions=[fused_direction])
        avg_loss = trainer.uniform_self_evaluate()
        with open("result.csv", "ab") as f:
            np.savetxt(f, avg_loss, comments="")
    end_time = time.time()
    print("total time {}".format(end_time-start_time))
