import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# seed random numbers to make calculation
# deterministic (just a good practice)
# np.random.seed(1)

# initialize weights randomly with mean 0
W = 2 * np.random.random((2, 1)) - 1

for iter in range(100000):
    args = np.array([[np.random.uniform(0, 0.5)], [np.random.uniform(0, 0.5)]])
    # forward propagation
    l1 = nonlin(np.dot(args.T, W))

    y = sum(args)
    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    W += np.dot(W, l1_delta)

print("Last data")
print(args)
print("Result:")
print(y)
print("NN Output:")
print(l1)

