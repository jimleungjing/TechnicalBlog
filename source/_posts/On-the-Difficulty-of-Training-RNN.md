---
title: The Issues on Training RNN
date: 2018-11-20 18:12:01
categories:
    - [Paper Notes]
tags:
    - [RNN]
mathjax: true
comments: true
---
<!--
1.       Problem: the problem to be solved, proposed by the author

2.       Solution: how the author solves the proposed problem

3.       Novelty: the difference from previous related work, and pick out the most related paper

4.       Take-away: what you learn from this paper and want to remember
-->
&emsp;In this post, we are going to briefly discuss the issues of training recurrent neural network(RNN)--the exploding and vanishing gradient problems. To learn further about the mechanism, we recommend reading the paper [On the difficulty of training recurrent neural networks, R. Pascanu et al. (2012)](https://arxiv.org/pdf/1211.5063.pdf) or [Learning Long-Term Dependencies with Gradient Descent is Difficult, Bengio, et al. (1994)](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) for a formal and detailed theory. And this post is mainly based on the former paper.
## Exploding and Vanishing Gradient
### RNN Model
&emsp;Using the notations in my another post: [The Computation of Gradients on RNN by BPTT](), also referred to the [*Deep Learning*](http://www.deeplearningbook.org), a RNN model with input $\boldsymbol{x}$ and hidden units $\boldsymbol{h}$ at time step $t$ is given by:
$$
\begin{align*}
\boldsymbol{h}^{(t)} &= \sigma(\boldsymbol{Ux}^{(t)}+\boldsymbol{Wh}^{(t-1)}+\boldsymbol{b}) \tag{1}
\end{align*}
$$ 
where the parameters $\boldsymbol{U}$, $\boldsymbol{W}$, $\boldsymbol{b}$ represent the input-to-hidden connections,hidden-to-hidden connections and bias vector respectively, and $\sigma$ is denoted as an activation function.


### Mechanism
&emsp;As a convenience, we here will take the weight matrix $\boldsymbol{W}$ for example to analyze the exploding and vanishing gradients problems. Introduced in [*Deep Learning*](http://www.deeplearningbook.org), variable $\boldsymbol{W}^{(t)}$, which is defined to be the copy of weight matrix $\boldsymbol{W}$ used only in time $t$, will be used in the computation of $\frac{\partial{L}}{\partial{\boldsymbol{W}}}$ by accumulating the factor $\frac{\partial{L}}{\partial{\boldsymbol{W}^{(t)}}} $ through time. The computation of gradient $\frac{\partial{L}}{\partial{\boldsymbol{W}}}$ can be written in a sum-of-products form as follows:

$$
\begin{align*}
\frac{\partial{L}}{\partial{\boldsymbol{W}}} &=
\sum_{1 \le t \le \tau}\frac{\partial{L}}{\partial{\boldsymbol{W}^{(t)}}} \tag{2}\\
\frac{\partial{L}}{\partial{\boldsymbol{W}^{(t)}}} &=
\sum_{t \le k \le \tau}\frac{\partial{L}^{(k)}}{\partial{\boldsymbol{h}^{(k)}}}
\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}}
\frac{\partial{\boldsymbol{h}^{(t)}}}{\partial{\boldsymbol{W}^{(t)}}} \tag{3} \\
\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}} &=
\prod_{t \lt i \le k}\frac{\partial{\boldsymbol{h}^{(i)}}}{\partial{\boldsymbol{h}^{(i-1)}}} = 
\prod_{t \lt i \le k}\boldsymbol{W}^{\mathsf{T}}\text{diag}(\sigma'(\boldsymbol{h}^{(i-1)})) \tag{4}
\end{align*}
$$
where $\sigma'$ computes element-wise the derivative of $\sigma$.
As the equation (3) shows, gradient term $\frac{\partial{L}}{\partial{\boldsymbol{W}^{(t)}}} $ consists of temporal components $\frac{\partial{L}^{(k)}}{\partial{\boldsymbol{h}^{(k)}}}
\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}}\frac{\partial{\boldsymbol{h}^{(t)}}}{\partial{\boldsymbol{W}^{(t)}}}$ that measures how $L$ at setp $k$ affects the gradient at step $t \lt k$. Notice that factor $\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}}$ take the form of a product of $k-t$ Jacobian matrices (see eq. (4)), naturally, we'd think it behaves in the same way a product of $k-t$ real number $n$ shrink to zero ($0\lt n \lt 1$) or explode to infinity ($n \gt 1$) when $k-t$ approaches to infinity. 
To formalize these intuitions, in what follows a method proposed by [R. Pascanu et al. (2012)](https://arxiv.org/pdf/1211.5063.pdf) will be used to obtain tight conditions for when the gradients explode or vanish.
At first, let's set $\sigma$ to the identity function in equation (1) so that the $\text{diag}(\sigma'(\boldsymbol{h}^{(i-1)}))$ will be an identity matrix $\boldsymbol{I}$. Consequently, equtaion (4) could be written in a product of $k-t$ matrices form:
$$
\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}} = 
\prod_{t \lt i \le k}\boldsymbol{W}^{\mathsf{T}} = 
\left(\boldsymbol{W}^{\mathsf{T}}
\right)^{(k-t)}   \tag{5}
$$
, and $\boldsymbol{W}$ can be further diagonalized to better understand the $k-t$th power of $\boldsymbol{W}$ if the n by n matrix $\boldsymbol{W}$ has n linearly indepentent eigenvectors:
$$
\begin{align*}
\boldsymbol{W} &= \boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1} \tag{6}\\
\left(
\boldsymbol{W}^{\mathsf{T}}
\right)^{(k-t)} &= 
\left(
\boldsymbol{S}\boldsymbol{\Lambda}^{(k-t)}\boldsymbol{S}^{-1}
\right)^{\mathsf{T}}    \tag{7}
\end{align*} 
$$
where $\boldsymbol{\Lambda}$ is the eigenvalue matrix and $\boldsymbol{S}$ is eigenvector matrix. Induced by eq. (7), it's sufficient for $0 \lt \rho \lt 1$, where the $\rho$ is the spectral radius of the weight matrix $\boldsymbol{W}$, for long term components to vanish (as $t \to \infty$) and necessary for $\rho \gt 1$ for them to blow up. 
Since the norm is used for matrices to measure the size of a matrix, we  now generalize the reasult above for nonlinear functions $\sigma$ where $|\sigma'(x)|$ is bounded, $\Vert\text{diag}(\sigma'(\boldsymbol{x}))\Vert \le \gamma \in \mathcal{R}$, by relying on singular values. 
We suppose that it is sufficient for $\lambda_1 \lt \frac{1}{\gamma}$, where $\lambda_1$ is the largest singular value of $\boldsymbol{W}$, for the vanishing gradient problem. Note that Jocabian matrix $\frac{\partial{\boldsymbol{h}^{(i)}}}{\partial{\boldsymbol{h}^{(i-1)}}}$ is given by $\boldsymbol{W}^{\mathsf{T}}\text{diag}(\sigma'(\boldsymbol{h}^{(i-1)}))$, we apply the triangle inequality of 2-norm of matrix to it:
$$
\forall i,\ \left\Vert\frac{\partial{\boldsymbol{h}^{(i)}}}{\partial{\boldsymbol{h}^{(i-1)}}}\right\Vert 
\le 
\left\Vert\boldsymbol{W}^{\mathsf{T}}\right\Vert \left\Vert\text{diag}(\sigma'(\boldsymbol{h}^{(i-1)})) \right\Vert
\lt
\frac{1}{\gamma}\gamma = 1  \tag{8}
$$
Let $\eta \in \mathbb{R}$ be such that $\forall i, \left\Vert\frac{\partial{\boldsymbol{h}^{(i)}}}{\partial{\boldsymbol{h}^{(i-1)}}} \right\Vert \le \eta \lt 1$, where the existence of $\eta$ is given by eq. (8). With this supremum, we have:
$$
\left\Vert
\frac{\partial{L}^{(k)}}{\partial{\boldsymbol{h}^{(k)}}}
\frac{\partial{\boldsymbol{h}^{(k)}}}{\partial{\boldsymbol{h}^{(t)}}}
\frac{\partial{\boldsymbol{h}^{(t)}}}{\partial{\boldsymbol{W}^{(t)}}}
\right\Vert 
\le
\eta^{k-t}
\left\Vert
\frac{\partial{L}^{(k)}}{\partial{\boldsymbol{h}^{(k)}}}
\right\Vert
\left\Vert
\frac{\partial{\boldsymbol{h}^{(t)}}}{\partial{\boldsymbol{W}^{(t)}}}
\right\Vert \tag{9}
$$
As the equation above shown, the long term contribution will go to 0 exponentially fast as $t-k$ grows because of $\eta \lt 1$. Similarly, we can get the necessary condition for the largest singular value $\lambda_1 \gt \frac{1}{\gamma}$, for the exploding gradient problem. For activation function sigmoid we have $\gamma = \frac{1}{4}$, for tanh we have $\gamma = 1$.

## Solutions to the vanishing and exploding gradient
### Scaling down the exploding gradient
Using L1 or L2 penalty on recurrent weight matrix is a good choice to deal with the exploding gradient problem. Compared with it, clipping the norm of matrix would be more simple and computationally efficient whenever it goes over a threshold although it introduce an additional hyper-parameter. 

### Vanishing gradient regularization
To address the vanishing gradient problem, [Hochreiter and Schmidhuber (1997)](https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735) propose the LSTM model that enforcing constant error flow through some special units. And [R. Pascanu et al. (2012)](https://arxiv.org/pdf/1211.5063.pdf) presented a regularization term so that gradients neither increase or decrease in magnitude. But frankly speaking, I can't catch the point what the regularization term excatly means in mathematics.

## Conclusion
In this post, I simply analyze the exploding and vanishing gradient problems in training RNNs by exploring the norm of gradients on weight matrices due to [R. Pascanu et al. (2012)](https://arxiv.org/pdf/1211.5063.pdf). The problems seem to be an obstacle keeping people from retaining information from the long time lags. Hopefully, many approaches were suggested to deal with them. For instance, it is universally acknowledged that LSTM models are widely applied in a varity of tasks and lead to the incredible successes. So, I'd like to explore the LSTMs in next post.   

