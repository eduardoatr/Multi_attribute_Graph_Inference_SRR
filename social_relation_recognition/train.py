import argparse

import objects.trainer as t


def main(args):

    trainer = t.Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')

    ###############
    ##   TRAIN   ##
    ###############
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--save', type=str, help='models to save', nargs='+', choices=[
        'max',
        'mean',
        'vote',
        'none'
    ])
    parser.add_argument('--epochs', type=int, help='min number of epochs')

    ###############
    ##   MODEL   ##
    ###############
    parser.add_argument('--arch', type=str, help='which architecture to use', choices=[
        'baseline',
        'edge_mlp'
    ])
    parser.add_argument('--classes', type=int, help='number of classes')
    parser.add_argument('--features', type=str, help='features to use', nargs='+', choices=[
        'bodies_activity',
        'bodies_age',
        'bodies_clothing',
        'bodies_gender',
        'context_activity',
        'context_emotion',
        'first_glance',
        'objects_attention'
    ])
    parser.add_argument('--time', type=int, help='number of time steps')
    parser.add_argument('--learning', type=float, help='initial learning rate')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--hidden', type=int, help='hidden state dimension')
    parser.add_argument('--loss', type=str, help='type of loss', choices=[
        'cross_entropy',
        'focal'
    ])
    parser.add_argument('--reduction', type=str, help='how to reduce losses', choices=[
        'mean',
        'sum',
        'none'
    ])
    parser.add_argument('--optimizer', type=str, help='type of optimizer', choices=[
        'adam',
        'adamw'
    ])
    parser.add_argument('--weight', type=float,
                        help='initial weight regularization for AdamW')
    parser.add_argument('--decay', type=float,
                        help='decay for learning rate and AdamW')
    parser.add_argument('--balance', type=str, help='balance samples by class', choices=[
        'weights_1',
        'weights_2',
        'none'
    ])
    parser.add_argument('--norm', type=str, help='apply node normalization', choices=[
        'true',
        'false'
    ])
    parser.add_argument(
        '--seed', type=str, help='int seed for reproducibility or pass the string random')
    parser.add_argument('--mode', type=str, help='run on eager or graph mode', choices=[
        'eager',
        'graph'
    ])

    ###############
    ##   PATHS   ##
    ###############
    parser.add_argument('--path_features', type=str, help='path to features')
    parser.add_argument('--path_save', type=str,
                        help='path to save the generated models or to load models')
    parser.add_argument('--path_results', type=str,
                        help='path to save tensorboard data')

    args = parser.parse_args()

    main(args)
