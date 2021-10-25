from plotter import Plotter
from trainer import Trainer
import tensorflow as tf
import numpy as np
import time
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
    # import pdb
    # pdb.set_trace()
    trainer.uniform_self_evaluate()

    # plotter = Plotter(trainer.model)
    # normalized_random_direction = plotter.create_random_direction(norm='layer')

    N = 625
    # step = 1/1000
    # # set init state
    # plotter.set_weights([normalized_random_direction], step=-step*N/2)

    # # plot N points in lossland
    start_time = time.time()
    for i in range(N):
    #     plotter.set_weights([normalized_random_direction], step=step)
        avg_loss = trainer.uniform_self_evaluate()
    #     with open("result_10000.csv", "ab") as f:
    #         np.savetxt(f, [avg_loss], comments="")
    end_time = time.time()
    print("total time {}".format(end_time-start_time))
