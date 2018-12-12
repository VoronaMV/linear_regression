import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Linear:

    def __init__(self, data: pd.DataFrame, target_name: str = ''):
        if not isinstance(data, pd.DataFrame):
            raise Exception('data argument should be pandas DataFrame instance')

        self.initial_dataframe = data
        self.col_names = data.columns

        if target_name not in data.columns:
            raise Exception(f'No such <{target_name}> column name in dataset')

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

    @property
    def coef(self) -> float:
        """
        Count coefficient of determination R^2.
        """
        Y_pred = self.X.dot(self.W)
        # sum of squares of regression residuals
        ss_res = np.sum((Y_pred - self.Y) ** 2)
        # sum of squares explained
        ss_reg = np.sum((Y_pred - self.Y.mean()) ** 2)
        # total sum of squares
        ss_total = ss_res + ss_reg
        return ss_reg / ss_total

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
                         coef=self.coef,
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
    def plot(train_data: pd.DataFrame, model_data: dict, predictions_df):
        independent_vars = model_data['independent_vars']
        dependent_var = model_data['dependent_var']
        X = train_data[independent_vars]
        _, cols = X.shape
        if cols > 1:
            raise Exception
        y = train_data[dependent_var]
        plt.xlabel(independent_vars[0])
        plt.ylabel(dependent_var)
        plt.scatter(X, y)
        plt.plot(predictions_df[independent_vars[0]], predictions_df[dependent_var])
        plt.show()
