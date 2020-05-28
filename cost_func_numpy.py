# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def gauss(x, mu, sig):
    scalor = 1 # / (sig * np.sqrt(2 * np.pi))
    f = scalor * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)
    return f


def line(x):
    y = x * 0.02 - 1
    return np.clip(y, -1, 0)


def cost(x):
    left = -gauss(x, 0, 5) - 0.20*gauss(x,-5,1)
    right = line(x)

    y = np.empty_like(x)
    mask_left = x < 0
    y[mask_left] = left[mask_left]
    y[~mask_left] = right[~mask_left]
    return y


x = np.linspace(-10, 10, 100)
y = cost(x)
plt.plot(x, y)
plt.show()

print("finished")
