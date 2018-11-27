import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pdb


class Test:

    def __init__(self, data: pd.DataFrame, independent_names=[], target_name: str = ''):
        if not isinstance(data, pd.DataFrame):
            raise Exception('data argument should be pandas DataFrame instance')

        self.initial_dataframe = data

        if independent_names and target_name:
            self.X = data[independent_names].values
            self.Y = data[target_name].values
        else:
            self.X = data.iloc[:, :-1].values
            self.Y = data.iloc[:, -1].values
        # add column with ones to independent var matrix
        self.Y = self.Y.reshape(len(self.Y), 1)

        self.raw_X = self.X

        self.X = self._stack_ones(self.X)
        _, cols = self.X.shape
        self.W = self.get_initial_weights(amount=cols, shape=(cols, 1))

    @staticmethod
    def _stack_ones(to_array: np.ndarray):
        rows_amount, _ = to_array.shape
        ones = np.ones(rows_amount).reshape(rows_amount, 1)
        return np.hstack((ones, to_array))

    @staticmethod
    def get_initial_weights(amount: int, shape: tuple) -> np.ndarray:
        flat_list = [0.0 for elem in range(amount)]
        array = np.array(flat_list, dtype=float).reshape(shape)
        return array

    def gradient_step(self, learning_rate: float, error: np.ndarray, rows: int, X: np.ndarray) -> np.ndarray:
        s = np.dot(error.T, X).T

        # s = X.T.dot(error)
        delta_W = 2 * (learning_rate * s / rows)
        return self.W - delta_W

    # def gradient_step(self, learning_rate: float, error: np.ndarray, rows: int, X: np.ndarray) -> np.ndarray:
    #     # pdb.set_trace()
    #     s = np.dot(error.T, X[:, 1]).T
    #     # s = X.T.dot(error)
    #     delta_W_b = 2 * (learning_rate * error[:, 0] / rows)
    #     delta_W_x = 2 * (learning_rate * s / rows)
    #     return self.W[0] - delta_W_b, self.W[1] - delta_W_x


    @staticmethod
    def cost(Y: np.ndarray, Y_pred: np.ndarray) -> float:
        rows, _ = Y.shape
        error = np.square(Y_pred - Y)
        mean_error = np.sum(error) / rows
        return mean_error

    def fit(self, learning_rate=0.000006, accuracy=0.0001):
        Y_pred = np.dot(self.X, self.W)
        cost0 = self.cost(self.Y, Y_pred)

        # i = 0
        while True:
            error = Y_pred - self.Y
            temp_W = self.W
            rows, cols = self.X.shape

            self.W = self.gradient_step(learning_rate, error, rows, self.X)

            Y_pred = np.dot(self.X, self.W)
            cost1 = self.cost(self.Y, Y_pred)
            current_accuracy = abs(cost1 - cost0)

            # if i == 0:
            #     i += 1
            #     continue

            if cost1 > cost0:
                self.W = temp_W
                print('bad')
                break
            elif current_accuracy <= accuracy:
                print('good')
                break

            cost0 = cost1

        return self.W

    def fit_sklearn(self):
        self.reg = LinearRegression().fit(self.raw_X, self.Y)
        print(self.reg.intercept_, self.reg.coef_)
        print(self.reg.predict(np.array([150500]).reshape(1, -1)))

    def predict(self, X: np.ndarray):
        return np.dot(X, self.W)


if __name__ == '__main__':
    data = pd.read_csv('data.csv', sep=',')

    # Normalize data
    # data = ((data - data.min()) / (data.max() - data.min()))

    linear = Test(data=data, independent_names=['km'], target_name='price')
    #weights = linear.fit(learning_rate=0.00000000006)
    weights = linear.fit(learning_rate=0.0001)
    print(weights.tolist())
    print(linear.predict(np.array([1, 22899]).reshape(1,2)).tolist())
    # linear.fit_sklearn()

# -0.02145*X + 8500