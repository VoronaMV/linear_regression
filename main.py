import pickle
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import argument_parser

class Linear:

    def __init__(self, data: pd.DataFrame, target_name: str = ''):
        if not isinstance(data, pd.DataFrame):
            raise Exception('data argument should be pandas DataFrame instance')

        self.initial_dataframe = data
        self.col_names = data.columns

        if target_name:
            independent_names = [col for col in data.columns if col != target_name]
            self.X = data[independent_names].values
            self.Y = data[target_name].values
            self.independent_names = independent_names
            self.dependent_name = target_name
        else:
            self.X = data.iloc[:, :-1].values
            self.Y = data.iloc[:, -1].values
            self.independent_names = self.col_names[:-1]
            self.dependent_name = self.col_names[-1]

        self.Y = self.Y.reshape(len(self.Y), 1)

        self.min_x = self.X.min(axis=0)
        self.max_x = self.X.max(axis=0)

        self.X = self._preprocess_data(self.X)

        self.X = self._stack_ones(self.X)

        _, cols = self.X.shape
        self.W = self.get_initial_weights(amount=cols, shape=(cols, 1))

    @staticmethod
    def _stack_ones(to_array: np.ndarray):
        rows_amount, _ = to_array.shape
        ones = np.ones(rows_amount).reshape(rows_amount, 1)
        return np.hstack((ones, to_array))

    @staticmethod
    def get_initial_weights(amount: int, shape: tuple, value=0.0) -> np.ndarray:
        flat_list = [value for _ in range(amount)]
        array = np.array(flat_list, dtype=float).reshape(shape)
        return array

    def gradient_step(self, learning_rate: float, loss: np.ndarray, rows: int, X: np.ndarray) -> np.ndarray:
        s = X.T.dot(loss)
        delta_W = 2 * (learning_rate * s / rows)
        return self.W - delta_W

    @staticmethod
    def _preprocess_data(X: np.ndarray) -> np.ndarray:
        normalized_X = ((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)))
        return normalized_X

    @staticmethod
    def cost(Y: np.ndarray, X :np.ndarray, theta: np.ndarray) -> float:
        rows, _ = Y.shape
        predictions = X.dot(theta)
        loss = predictions - Y
        mean_sqr_error = np.sum(loss ** 2) / rows
        return mean_sqr_error

    @staticmethod
    def to_file(filename='result', **kwargs):
        with open(filename, 'wb') as file:
            pickle.dump(kwargs, file)

    @staticmethod
    def from_file(filename='result') -> dict:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    def fit(self, learning_rate=0.1, accuracy=0.0000000001, to_file='result'):
        Y_pred = np.dot(self.X, self.W)
        cost_prev = self.cost(self.Y, self.X, self.W)
        rows, cols = self.X.shape

        while True:
            loss = Y_pred - self.Y
            temp_W = self.W

            self.W = self.gradient_step(learning_rate, loss, rows, self.X)

            Y_pred = np.dot(self.X, self.W)
            cost_curr = self.cost(self.Y, self.X, self.W)
            current_accuracy = abs(cost_curr - cost_prev)

            if cost_curr > cost_prev:
                self.W = temp_W
                break
            if current_accuracy <= accuracy:
                break
            cost_prev = cost_curr
        if to_file:
            self.to_file(filename=to_file,
                         weights=self.W,
                         min_x=self.min_x,
                         max_x=self.max_x,
                         independent_vars=self.independent_names,
                         dependent_var=self.dependent_name,
                         initial_X=self.X[:, 1:],
                         x_shape=self.X[:, 1:].shape,
                         Y=self.Y)

        return self.W

    @classmethod
    def predict(cls, X: np.ndarray, model_data):

        W = model_data['weights']
        min_x = model_data['min_x']
        max_x = model_data['max_x']

        try:
            rows, cols = X.shape
        except ValueError:
            cols, *_ = X.shape
            rows = 1
        X = X.reshape(rows, cols)
        X = ((X - min_x) / (max_x - min_x))
        X = cls._stack_ones(X)
        return np.dot(X, W)

    @staticmethod
    def plot(train_data: pd.DataFrame, model_data: dict, predictions_df, ):

        independent_vars = model_data['independent_vars']
        dependent_var = model_data['dependent_var']
        X = train_data[independent_vars]
        print(X.shape)
        _ , cols = X.shape
        if cols > 1:
            raise Exception
        y = train_data[dependent_var]
        plt.xlabel(independent_vars[0])
        plt.ylabel(dependent_var)
        plt.scatter(X, y)
        plt.plot(predictions_df[independent_vars[0]], predictions_df[dependent_var])
        plt.show()


def handle_result(is_print=True, to_file=True, **kwargs):
    df = pd.DataFrame(kwargs).astype(int)
    if to_file:
        df.to_csv('prediction_result.csv')
    if is_print:
        print(df)

    return df


def print_formula(weights: np.ndarray):
    weights = weights.flatten().tolist()
    formula = f'y={weights[0]}'

    formula += ''.join(f'+{round(weight, 3)}*X{i}'
                       if weight > 0 else f'{round(weight, 3)}*X{i}'
                       for i, weight
                       in enumerate(weights[1:]))
    print('FORMULA:', formula)


if __name__ == '__main__':

    args = argument_parser()

    if not args.predict:
        if not os.path.isfile(args.file):
            exit(f'No such file {args.file}')
        data = pd.read_csv(args.file, sep=',')
        linear = Linear(data=data,
                        target_name=args.target_var)
        weights = linear.fit(learning_rate=args.learning_rate,
                             accuracy=args.accuracy,
                             to_file=args.to_file)
    else:
        if not os.path.isfile(args.model_file):
            exit(f'No such file {args.model_file}')
        if args.predict_from_file:
            if not os.path.isfile(args.predict_from_file):
                exit(f'No such file {args.predict_from_file}')
            data = np.loadtxt(args.predict_from_file, delimiter=',')
        else:
            data = []
            while True:
                value = input('Write digit or q: ')
                if value in 'Qq':
                    if not data:
                        continue
                    break
                else:
                    if not value.isdigit():
                        continue
                    data.append(int(value))
            data = np.array(data)

        try:
            model_data = Linear.from_file(args.model_file)
            x_shape = model_data['x_shape']
            if data.shape != x_shape:
                data = data.reshape(len(data // len(model_data['independent_vars'])), x_shape[1])
            prediction = Linear.predict(data, model_data)
            print_formula(model_data['weights'].flatten())
            params = {
                model_data['independent_vars'][0]: data.flatten(),
                model_data['dependent_var']: prediction.flatten()
            }
            print(params)
            result_df = handle_result(**params)
        except:
            exit('Bad data')

        if args.plot:
            if not os.path.isfile(args.file):
                exit(f'No such file {args.file}')

            train_data = pd.read_csv(args.file, sep=',')
            Linear.plot(train_data, model_data, predictions_df=result_df)

# -0.02145*X + 8500
