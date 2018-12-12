import argparse
import pandas as pd
import numpy as np


def handle_result(is_print=True, to_file=True, **kwargs):
    df = pd.DataFrame(kwargs).astype(int)
    if to_file:
        df.to_csv('prediction_result.csv')
    if is_print:
        print(df)
    return df


def print_formula(weights: np.ndarray, coef=None):
    weights = weights.flatten().tolist()
    formula = f'y={weights[0]}'

    formula += ''.join(f'+{round(weight, 3)}*X{i}'
                       if weight > 0 else f'{round(weight, 3)}*X{i}'
                       for i, weight
                       in enumerate(weights[1:]))
    print('FORMULA:', formula)
    if coef:
        print(f'Coefficient of determination: {round(coef, 2)}')


def argument_parser():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-P', '--predict',
                        action='store_true',
                        help="Flag if you want to run prediction mode. Should be used with -MD flag",
                        dest='predict')
    parser.add_argument('-PFL', '--predict-from-file',
                        help="Set file destination with data to be predicted",
                        dest='predict_from_file')
    parser.add_argument('-MD', '--model',
                        help="Put destination of file with trained model",
                        dest='model_file',
                        default='result')
    parser.add_argument("-PLT", "--plot",
                        action='store_true',
                        dest='plot',
                        help="Plot data with linear function. You should have file with trained model."
                             " Should be used with -F flag.")
    parser.add_argument('-F', '--file',
                        dest='file',
                        default='',
                        help='Destination File with train data. Mandatory param.')
    parser.add_argument('-TO', '--tofile',
                        dest='to_file',
                        default='result',
                        help='Destination File with trained model data. Default is ./result')
    parser.add_argument('-LR', '--learning-rate',
                        # type=str,
                        dest='learning_rate',
                        type=float,
                        default=0.1,
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
