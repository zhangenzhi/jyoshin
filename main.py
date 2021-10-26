import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from plotter import Plotter
from trainer import Trainer



if __name__ == '__main__':
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 100, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh'], 'fuse_models': None},
                    }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    trainer.uniform_self_evaluate()

    plotter_args = {'num_evaluate': 10000,
                    'step': 1/1000,
                    'fuse_models': trainer_args['model']['fuse_models'],
                    }
    plotter = Plotter(plotter_args, trainer.model)

    # set init state
    normalized_random_direction = plotter.create_random_direction(norm='layer')
    plotter.set_weights([normalized_random_direction], init_state=True)

    # plot num_evaluate*fuse_models points in lossland
    start_time = time.time()
    for i in range(plotter.num_evaluate):
        plotter.set_weights([normalized_random_direction])
        avg_loss = trainer.uniform_self_evaluate()
        with open("result_10000.csv", "ab") as f:
            np.savetxt(f, avg_loss, comments="")
    end_time = time.time()
    print("total time {}".format(end_time-start_time))
