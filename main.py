import optimize
import cost_func_torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.gridspec

# change default colors
import matplotlib as mpl
from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def main():
    # define params
    # loss_func = lambda x: x.pow(2)
    loss_func = cost_func_torch.cost
    lst_x_start = [10]  # , -5, -10, -7]  # [-10, -7, -5, -3, -2.5, -1, 5, 10]
    lst_optimizer_name = ['sgd', 'adagrad', 'rmsprop', 'adam1', 'adam2', 'adam_default']
    nsteps_max = 5000

    for x_start in lst_x_start:
        print("Working on x_start={}".format(x_start))
        lst = []
        for optimname in lst_optimizer_name:
            # run optimization
            optimizer, optimizer_kwargs = get_optimizer(optimname)
            xs, losses = optimize.optimize(loss_func,
                                           optimizer=optimizer,
                                           optimizer_kwargs=optimizer_kwargs,
                                           x_start=x_start,
                                           nsteps_max=nsteps_max
                                           )
            lst.append((optimname, optimizer_kwargs, xs, losses))

        # plot and save
        title = "xstart_{:.3f}".format(x_start)
        fig, axes = plot(lst, loss_func)
        axes[1].set_ylim([nsteps_max, 0])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig("output/" + title + ".png")
        plt.close(fig)
    # plt.show()


def get_optimizer(name):
    lr = 0.005
    alpha = 0.99
    dct = {
        'adadelta': (torch.optim.Adadelta, {'lr': lr}),
        'adagrad': (torch.optim.Adagrad, {'lr': lr * 10}),  # if not optim.zero, *10 ?!
        'adam1': (torch.optim.Adam, {'lr': lr, 'betas': (0, alpha)}),  # default 0.9, 0.999
        'adam2': (torch.optim.Adam, {'lr': lr, 'betas': (alpha, alpha)}),  # default 0.9, 0.999
        'adam_default': (torch.optim.Adam, {'lr': lr, 'betas': (0.9, 0.999)}),  # default 0.9, 0.999
        'adamw': (torch.optim.AdamW, {'lr': lr}),
        'rmsprop': (torch.optim.RMSprop, {'lr': lr, 'alpha': alpha}),  # default 0.99
        'lbfgs': (torch.optim.LBFGS, {'lr': lr}),
        'sgd': (torch.optim.SGD, {'lr': lr * 50}),  # * 50 in linear case!, since 0.02 line slope!
    }
    return dct[name]


def plot(lst, loss_func, xmin=-10, xmax=10):
    # create figure
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.get_shared_x_axes().join(ax1, ax2)

    # plot cost function
    xs = torch.tensor(np.linspace(xmin, xmax, 1000), requires_grad=False)
    losses = [loss_func(x) for x in xs]
    ax1.plot(xs, losses, 'k')
    ax1.set_ylabel('loss')
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )

    # plot x's along way
    linestyles = ['solid', 'dashdot', 'dashed', 'dotted', 'dotted', 'dotted']
    for cnt, (optimname, optimizer_kwargs, xs, losses) in enumerate(lst):
        time = range(len(xs))
        label = optimname + ":" + ",".join(["{}-{}".format(key, val) for key, val in optimizer_kwargs.items()])
        ax2.plot(xs, time,
                 linestyle=linestyles[cnt % len(linestyles)],
                 label=label,
                 )

    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('time in optimization steps')

    # make grid
    ax1.set_xlim([xmin, xmax])
    ax1.xaxis.grid(True)
    ax2.xaxis.grid(True)
    fig.tight_layout()

    return fig, [ax1, ax2]


if __name__ == '__main__':
    main()
