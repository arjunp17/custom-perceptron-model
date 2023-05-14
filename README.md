# The Perceptron model

For data $\bigl(x_i, y_i\bigr)_{i=1}^N$ with inputs $x_i \in \mathbb{R}^d$ and $y_i \in \bigl(1, -1\bigr)$, the perceptron model is defined as,

$f(x) = 1$, if  $W^Tx + b  >= 0$

$f(x) = -1$, if  $W^Tx + b  < 0$

where $W \in \mathbb{R}^d$ and $b \in \mathbb{R}$ are the parameters of the model.

# Custom Learning Algorithm

Given a minibatch of data $\bigl(x_i, y_i\bigr)_{i=1}^M$, and the model prediction as $\hat{y_i} = f(x_i)$, the model weight-update rule is defined by

$$W:= \frac{1}{M}\sum_{i=1}^{M}\biggl(W + \bigl(y_{i} - \hat{y}_{i}\bigl)x_i/2\biggl)$$

$$b:= \frac{1}{M}\sum_{i=1}^{M}\biggl(b + \bigl(y_{i} - \hat{y}_{i}\bigl)/2\biggl)$$

where $M$ is the mini batch size.
