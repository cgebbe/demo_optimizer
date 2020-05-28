# import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt


def gauss(x, mu, sig):
    scalor = 1  # / (sig * np.sqrt(2 * np.pi))
    f = scalor * torch.exp(-0.5 * (x - mu) ** 2 / sig ** 2)
    return f


def line(x):
    y = x * 0.02
    return torch.clamp(y, min=0)


def cost(x):
    if x < 0:
        return 1 - gauss(x, 0, 5) - 0.19 * gauss(x, -5, 1)
    else:
        return line(x)

if __name__ == '__main__':
    x = torch.linspace(-10, 10, 100)
    y = [cost(xx) for xx in x]
    plt.plot(x, y)
    plt.show()
    print("finished")
