import argparse


def argument_parser():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-P', '--predict',
                        #type=bool,
                        #default=False,
                        action='store_true',
                        help="Flag if you want to run prediction mode",
                        dest='predict')
    parser.add_argument('-PFL', '--predict-from-file',
                        help="Flag if you want to run prediction mode",
                        dest='predict_from_file')
    parser.add_argument('-MD', '--model',
                        #type=str,
                        # action='store_true',
                        help="Put destination of file with trained model",
                        dest='model_file',
                        default='result')
    parser.add_argument("-PLT", "--plot",
                        action='store_true',
                        dest='plot',
                        help="Plot data with linear function. You should have file with trained model")
    parser.add_argument('-F', '--file',
                        #type=str,
                        dest='file',
                        default='',
                        #action='store_true',
                        help='Destination File with train data. Default is ./data.csv')
    parser.add_argument('-TO', '--tofile',
                        #type=str,
                        dest='to_file',
                        default='result',
                        #action='store_true',
                        help='Destination File with trained model data. Default is ./result')
    parser.add_argument('-LR', '--learning-rate',
                        # type=str,
                        dest='learning_rate',
                        type=float,
                        default=0.1,
                        # action='store_true',
                        help='Learning rate for gradient descent step')
    parser.add_argument('-A', '--accuracy',
                        type=float,
                        dest='accuracy',
                        default=0.0000000001,
                        help='Accuracy - difference between cost functions of previous and current step')
    parser.add_argument('-T', '--target-variable',
                        dest='target_var',
                        type=str,
                        default='price',
                        help='Target (dependent) variable name to predict')

    args = parser.parse_args()
    return args
