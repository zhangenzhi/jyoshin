from utils import *
import argparse
from trainer import UniformTrainer, Cifar10Trainer
from plotter import Plotter
from label_generator import generate_label_for_cifar10
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot lossland around ture minima.')
    parser.add_argument(
        '--config', default='scripts/configs/1d_uniform_dnn.yaml')

    args = parser.parse_args()
    config = get_yml_content(args.config)

    trainer_args = config['Trainer']
    plotter_args = config['Plotter']

    if trainer_args['dataset']['name'] == 'uniform':
        trainer = UniformTrainer(trainer_args)
    elif trainer_args['dataset']['name'] == 'cifar10':
        trainer = Cifar10Trainer(trainer_args)
        # trainer.run()

    weights_trajectory = []
    for i in range(trainer_args['dataset']['epoch']):
        weights_trajectory.append(
            trainer.load_weights_trajectory(index=i,
                    filepath=trainer_args['others']['save_trajectory'])
            )
        
    import pdb
    pdb.set_trace()

    # generate_label_for_cifar10(dataset=iter(trainer.plotter_dataset),
    #                            model=trainer.model,
    #                            path_to_file=plotter_args['path_to_adapt_label'])

    # plotter = Plotter(plotter_args, trainer)
    # plotter.run()
