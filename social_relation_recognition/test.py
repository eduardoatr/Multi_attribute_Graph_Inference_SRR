import argparse

import objects.tester as t


def main(args):

    tester = t.Tester(args)
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')

    ###############
    ##   TEST    ##
    ###############
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--type', type=str, help='model type to load', choices=[
        'max',
        'mean',
        'vote'
    ])
    parser.add_argument('--metric', type=str, help='model metric to load', choices=[
        'precision',
        'accuracy',
        'accuracy'
    ])

    ###############
    ##   PATHS   ##
    ###############
    parser.add_argument('--path_features', type=str, help='path to features')
    parser.add_argument('--path_configs', type=str,
                        help='path to model configs')
    parser.add_argument('--path_models', type=str, help='path to saved models')

    args = parser.parse_args()

    main(args)
