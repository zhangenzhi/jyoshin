import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer import Trainer
from plotter import Plotter
import tensorflow as tf
import numpy as np
import time


if __name__ == '__main__':
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 100, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh'], 'fuse_models': 3000},
                    }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    # import pdb
    # pdb.set_trace()
    trainer.uniform_self_evaluate()


    N = 600
    fuse_models = trainer_args['model']['fuse_models']
    step = 1/(1000*fuse_models)
    plotter = Plotter(trainer.model,fuse_models=fuse_models)

    # # set init state
    normalized_random_direction = plotter.create_random_direction(norm='layer')
    plotter.set_weights([normalized_random_direction],
                        step=-step*N*fuse_models/2)

    # # plot N points in lossland
    start_time = time.time()
    for i in range(N):
        plotter.set_weights([normalized_random_direction],
                            step=step)
        avg_loss = trainer.uniform_self_evaluate()
    #     with open("result_10000.csv", "ab") as f:
    #         np.savetxt(f, [avg_loss], comments="")
    end_time = time.time()
    print("total time {}".format(end_time-start_time))
