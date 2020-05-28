# About

Readme

# Definitions of optimizers

We define gradient $g_x(t)=\frac{\partial}{\partial x} l(t)$.

### Stochastic Gradient Descent (SGD)

formula
$$
dx(t)= - \eta \partial_x l(t)
$$

### SGD with momentum

$$
dx(t)= - \eta\partial_x l(t) + \alpha dx(t-1)
$$

, where $\alpha\in[0,1]$ is a decay factor

### Adaptive Gradient algorithm (AdaGrad)

$$
v(t)=\sum_{\tau=0}^{t-1} \left( \partial_x l(\tau) \right)^2
\\
dx(t) = -\frac{1}{\sqrt{v(t) }} \eta\partial_x l(t)
$$

- Denominator will only increase in time and thus dampen step size. Makes sense for convex problems but not that much for non-convex problems.
- Denominator magnitude will be small for sparse parameters or parameters receiving only small updates, so that spare parameters will still get updated after time.

### RMSProp

$$
v(t) = \alpha v(t-1) + (1-\alpha)(\partial_x l(t))^2
\\
dx(t) = -\frac{1}{\sqrt{v(t)}}\eta\partial_x l(t)
$$

- Similarly to AdaGrad, denominator dampens step size. Contrary to AdaGrad, denominator does not only increase, but can also increase! Simply running average of gradients, instead of cumulative sum
- **Important**: If $\partial_x l(t)$ is approximately constant $\forall t$, we get  $\sqrt{v(t)}\rightarrow |\partial_x l(t)|$ and thus  $dx=-\eta sign(\partial_x l(t))$. This means a constant absolute size independent of the gradient.
- See a great explanation: https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a

### Adaptive Moment Estimation (Adam)

$$
v(t)= \frac{1}{1-\alpha^t} \cdot \left[\alpha v(t-1) + (1-\alpha) \left( \partial_x l(t) \right)^2 \right] = E[(\partial_x l(t))^2]
\\
g(t) = \frac{1}{1-\beta^t} \cdot \left[ \beta g(t-1) + (1-\beta)\partial_x l(t) \right] = E[\partial_x l(t)]
\\
dx(t) = -\frac{1}{\sqrt{v(t)}}\eta g(t)
$$

, where $\alpha\in[0,1]$ and $\beta\in[0,1]$ .

- Denominator very similar to RMSProp, but correction term yields larger values at the beginning
- Gradient is also 
- Benefit: If there is a (brief) hill, it should pass through, or?!





# Conclusions with examples

### If the cost is linear, the step size of all algorithms is constant over time

XXX

### RMSProp and Adam have a unit step size for small $\alpha$ 

XXX