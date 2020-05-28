# About

Readme

# Report

### Notation

We use the following notation:

- $dx(t)$ step at time t . Used for calculating $x(t+1) = x(t) + dx(t)$
- $l(t) =l(x(t))$ = loss at time t
- $\partial_x l(t) = \frac{\partial}{\partial x} l(x(t))$ = derivative of loss with respect to x (at time t)
- $\eta$= learning rate (constant parameter set by user)

Note that we only consider a one-dimensional (scalar) parameter $x$ in order to simplify the discussion and visualization.

### Stochastic Gradient Descent (SGD)

The simplest optimizer is the stochastic gradient descent, which is defined as 
$$
dx(t)= - \eta \partial_x l(t)
$$

It will serve as our base optimizer for comparison.

For example, in the image below the loss function is a simple linear function $l(x)=1.0*x$. For linear functions the gradient $\partial_x l(t)=1.0$ is constant and thus also the step size is constant $dx=-\eta\partial_x l(t) = -\eta \cdot 1.0 = -\eta$.  If we start at $x=10$ and have a learning rate  of $\eta=0.005$, we need $nsteps=x/dx = 10/0.005 = 2000$ steps to reach the minimum at  $x=0$.

![](output/sgd_slope1.png)



### Adaptive Gradient algorithm (AdaGrad)

AdaGrad is defined as followed
$$
v(t)=\sum_{\tau=0}^{t-1} \left( \partial_x l(\tau) \right)^2
\\
dx(t) = -\frac{1}{\sqrt{v(t) }} \eta\partial_x l(t)
$$

One characteristic of AdaGrad is that the step size reduces over time, since the denominator will only increase over time. This dampening effect can be seen in the image below. Such a damping may make sense for convex problems where we want to start off with large step sizes and take smaller steps once we reach the optimum. For nonconvex problems, this dampening is likely undesired.

![](output/adagrad_slope1.png)



A second characteristic of AdaGrad is that it is robust against scaling of the loss function, meaning the step size does not change if we scale the loss function with any constant. This can be very easily seen if the gradient  $\partial_x l(t)\approx k$ is approximately constant over time. Then, the equation simplifies to 
$$
v(t)=\sum_{\tau=0}^{t-1} \left( \partial_x l(\tau) \right)^2 \approx \sum_{\tau=0}^{t-1} k^2 = t\cdot k^2
\\
dx(t) = -\frac{1}{\sqrt{v(t) }} \eta\partial_x l(t) \approx -\frac{1}{\sqrt{t}|k|}\eta k = -\frac{1}{\sqrt{t}}\eta sign(k)
$$
which shows that the step size is independent of the loss function! The image below shows again a linear line but this time with slope $0.5$ instead of $1.0$.  While SGD now takes twice as long to find the minimum, AdaGrad isn't affected by the scaling at all (compare with image above).

![](output/adagrad_slope0.5.png)

A third very important characteristic of AdaGrad is that the denominator $v(t)$ works for every parameter individually. For example, if we would not only have one parameter $x$ to optimize, but multiples $\vec{x}=[x1,x2]$, the scaling $v(t)$ would be independent for each parameter. This is very beneficial in case of sparse data: The step size of sparse parameters will dampen much slower than the step size of frequent parameters.



### Root Mean Square Propagation (RMSProp)

RMSProp can be defined as followed
$$
v(t) = \alpha v(t-1) + (1-\alpha)(\partial_x l(t))^2 \quad \text{ with } \alpha\in[0,1] 
\\
dx(t) = -\frac{1}{\sqrt{v(t)}}\eta\partial_x l(t)
$$

Similarly to AdaGrad, the step size is divided by a denominator. Contrary to AdaGrad, this denominator is not the cumulative sum which only increases, but instead an [exponential moving average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) of the squared gradients. Therefore, the step size does not continually decrease like in Adagrad, but is rather approximately constant for a linear slope, see image below.

![](output/rmsprop_slope1.png)

Similar to AdaGrad, if $\partial_xl(t)=k$ is constant, we get  $v(t)\rightarrow k^2$ and thus  $dx=-\eta sign(k)$, implying robustness against scaling of the loss function as shown in the image below

![](output/rmsprop_slope0.5.png)

The approximately constant step size is a very desired property in case of saddle points, where the gradient becomes very small. SGD will spent a lot of time on such saddle points, while RMSProp is able to overcome them quickly, see image below. The slope in the saddle is only $0.1$ compared to a slope of $1$ everywhere else.

![](output/rmsprop_saddle.png)

A closeup of the saddle point (see image below) reveals, that the step size in RMSProp is increasing the longer it stays in the saddle point. If the saddle would be longer, the step size would again reach  $dx=-\eta sign(k)$ as before the saddle point. Another interesting observation is that RMSProp's step size after the saddle point ($x<5$) is even larger than the one before the saddle point ($x<6$). This is because directly after the saddle point the denominator $v(t)\approx 0.1 ^2$ is smaller than before the saddle point $v(t)\approx 1^2$, while the gradient $\partial_x l(t)=1.0$ is equal.

![](output/rmsprop_saddle_detail.png)



### Adaptive Moment Estimation (Adam)

Adam uses the same denominator $v(t)$ as RMSProp, but multiplied with the bias correction term $1/(1-\alpha^t)$ . Moreover, it also employs an exponential moving average of the gradient $\partial_x l(t)$. 
$$
v(t)= \frac{1}{1-\alpha^t} \cdot \left[\alpha v(t-1) + (1-\alpha) \left( \partial_x l(t) \right)^2 \right] \quad \text{ with } \alpha\in[0,1]
\\
g(t) = \frac{1}{1-\beta^t} \cdot \left[ \beta g(t-1) + (1-\beta)\partial_x l(t) \right]  \quad \text{ with } \beta\in[0,1]
\\
dx(t) = -\frac{1}{\sqrt{v(t)}}\eta g(t)
$$

Similar to RMSprop, the step size is robust to a scaling of the loss function and approximately constant for a  linear slope. In fact, for a linear slope, Adam's step is exactly constant and exactly equals SGD's step for slope $1.0$ (see image below). While RMSprop's needs a slow ramp-up phase (the first ~50 steps) to reach a constant step size, Adam's step size is constant from the beginning. This is exactly the effect of the bias correction terms $1/(1-\alpha^t)$ and $1/(1-\beta^t)$. It prevents the ramp-up phase visible for RMSProp and thus accelerates the optimization process.

![](output/adam_slope1.png)

![](output/adam_slope0.5.png)

The behavior in saddle points is also very similar to RMSProp. Adam's exact behavior depend a lot on the parameters $\alpha$ and $\beta$, so the images below show two different settings.

![](output/adam_saddle1.png)

![](output/adam_saddle2.png)

The most important difference compared to RMSProp can be seen in case of local minima (around $x=2.17$ in the image below). SGD, Adagrad and RMSProp will get stuck in such local minima. In contrast, Adam is able to overcome those (up to a certain degree). 

![](output/adam_spike.png)
![](output/adam_spike_detail.png)

There is a variant of RMSProp called RMSProp with momentum, which is also able to overcome local minima. A detailed comparison between RMSProp with momentum and Adam is presented in the following thread: https://datascience.stackexchange.com/questions/26792/difference-between-rmsprop-with-momentum-and-adam-optimizers/39224

