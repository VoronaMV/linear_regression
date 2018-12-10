import numpy as np


def cost(Y, X, theta):
    predictions = X.dot(theta)
    loss = predictions - Y
    sum_sqr_error = np.sum(loss ** 2) / len(loss)
    return sum_sqr_error


def gradient(Y, X, theta, learning_rate=0.1):
    predictions = np.dot(X, theta)
    loss = predictions - Y
    tmp = X.T.dot(loss)
    # tmp = np.dot(loss.T, X).T
    delta_theta = learning_rate * tmp / 2 / len(loss)
    return theta - delta_theta

# ((X - X.min()) / (X.max() - X.min()))
if __name__ == '__main__':
    data = np.loadtxt('data.csv', skiprows=1, delimiter=',')

    theta = np.array([0.0, 0.0]).reshape(2, 1)

    initial_X = data[:, 0].reshape(len(data), 1)
    tmp_X = ((initial_X - initial_X.min()) / (initial_X.max() - initial_X.min()))
    ones = np.ones(data.shape[0]).reshape(len(data), 1)
    X = np.hstack((ones, tmp_X))

    Y = data[:, 1].reshape(len(data), 1)
    # Y = ((Y - Y.min()) / (Y.max() - Y.min()))

    cost0 = cost(Y, X, theta)

    for i in range(1000000000):

        tmp_theta = theta
        theta = gradient(Y, X, theta)

        cost1 = cost(Y, X, theta)

        if cost1 > cost0:
            print(abs(cost1 - cost0))
            break
        elif abs(cost1 - cost0) <= 0.00000000001:
            print('yep')
            break
        elif np.array_equal(tmp_theta, theta):
            print('yep')
            break

        # print(abs(cost1 - cost0))

        cost0 = cost1
        # print(theta.flatten())

    print(theta)

    test_x = (22899 - initial_X.min()) / (initial_X.max() - initial_X.min())
    print('test_x', test_x)
    test = np.array([1, test_x])
    print(test.dot(theta))
