---
title: The Computation of Gradients on RNN by BPTT
date: 2018-11-8 16:53:42
categories: 
    - [Deep Learning]
tags: 
    - [RNN]
    - [BPTT]
mathjax: true
comments: true
---
&emsp;As a beginner of deep learning, it's quite easy to get lost when you attempt to compute the gradient on a recurrent neural network(RNN). Therefore, we will give the computation of gradients on RNN by backforward propagation through time(BPTT). I assume that you have known the concept of backward-propagation algorithm, [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network)(RNN), [linear algebra](http://math.mit.edu/~gs/linearalgebra/) and [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus) before the exploration.
## Notations
&emsp;These notations used in the following computation refer to the book [*Deep Learning*](http://www.deeplearningbook.org).
$
\begin{array}{l|l}\hline
a & \text{Scalar a}\\\hline
\boldsymbol{x} & \text{Vector x} \\\hline
\boldsymbol{1}_{n} & \text{A vector [1,1,...,1] with n column}\\\hline
\boldsymbol{A} & \text{Matrix A}\\\hline
\boldsymbol{A}^{-1} & \text{The inverse matrix of matrix A}\\\hline
\boldsymbol{A}^{\mathsf{T}} & \text{Transpose of matrix A}\\\hline
\text{diag}(\boldsymbol{a}) & \text{Diagnal matrix with diagnal entries gieven by }\boldsymbol{a}\\\hline
\boldsymbol{I} & \text{Identity matrix}\\\hline
\nabla_{\boldsymbol{x}}y & \text{Gradient }y\text{ with respect to }\boldsymbol{x} \\\hline
\frac{\partial{y}}{\partial{x}} & \text{Partial derivative of } y \text{ with respect to }x\\\hline
\frac{\partial{\boldsymbol{y}}}{\partial{\boldsymbol{x}}} & \text{Jacobian matrix }\boldsymbol{J} \in \mathbb{R}^{m \times n} \text{of } f:\mathbb{R}^{n}\to\mathbb{R}^{m} \\\hline
\end{array}
$

## Review of RNN 
<!--
在align环境中根据'&'对齐文本
-->
<center><img src="BPTT/RNN.png" alt="The computational graph of RNN" width=70% length=70%></center>
<p style="text-align:center">
Figure 2.1: The computational graph of RNN @<i><a href=http://www.deeplearningbook.org>Deep Leaerning</a></i>
</p>
&emsp;Let's develop the forward propagation equations for the RNN dipcted in Figure 2.1. As a convenience, we use right superscription to indiacte the specified state of RNN at some time $ t $. 
To represent the hidden units of the network, we use the variable $ \boldsymbol{h} $, which will be the input of hyperbolic tangent activation function. As you see, the RNN has input-to-hidden connections parametrized by a weight matrix $ \boldsymbol{U} $, hidden-to-hidden connections parametrized by a weight matrix $ \boldsymbol{W} $, hidden-to-output connections parametrized by a weight matrix $ \boldsymbol{V} $. We can apply the softmax operation to the output $ \boldsymbol{o} $ to obtain a vector $ \boldsymbol{\hat{y}} $ of normalized probabilities over the output. We define the loss function as the negative log-likelihood of output vector $ \boldsymbol{\hat{y}} $ given the training target $ \boldsymbol{y} $. With notations and definitions made, we have the following equations:
$$
\left\{
\begin{aligned}
\boldsymbol{a}^{(t)} &= \boldsymbol{b} + \boldsymbol{Wh}^{(t-1)} + \boldsymbol{Ux}^{(t)} &\qquad (2.1)\\
\boldsymbol{h}^{(t)} &= \text{tanh}(\boldsymbol{a}^{(t)}) &\qquad (2.2)\\
\boldsymbol{o}^{(t)} &= \boldsymbol{c} + \boldsymbol{Vh}^{(t)} &\qquad (2.3)\\
\boldsymbol{\hat{y}}^{(t)} &= \text{softmax}(\boldsymbol{o}^{(t)}) &\qquad (2.4)\\
L^{(t)} &= -{\boldsymbol{y}^{(t)}}^{\mathsf{T}}\text{log}(\boldsymbol{\hat{y}}^{(t)}) &\qquad (2.5)
\end{aligned}
\right.
$$
where both $ \text{tanh} $ and $\text{log}$ are element-wise function.

## Taking the Derivatives
&emsp;Known the notations and concepts above, we can compute the gradients by BPTT using the [identities](https://en.wikipedia.org/wiki/Matrix_calculus#Identities) below.

| Conditions | Expression | Denominator layout |
| :--: | :--: | :--: |
| $$ a = a(\boldsymbol{x}), \boldsymbol{u = u}(\boldsymbol{x}) $$ | $$ \frac{\partial{a\boldsymbol{u}}}{\partial{\boldsymbol{x}}} $$ | $$ a\frac{\partial{\boldsymbol{u}}}{\partial{\boldsymbol{x}}} + \frac{\partial{a}}{\partial{\boldsymbol{x}}}\boldsymbol{u}^{\mathsf{T}} $$ |
| $ \boldsymbol{A} $ is not a function of $ \boldsymbol{x} $ | $ \frac{\partial{\boldsymbol{Ax}}}{\partial{\boldsymbol{x}}} $ | $ \boldsymbol{A}^{\mathsf{T}} $  |
| $g(\boldsymbol{U})\text{ is a scalar, }\boldsymbol{U} = \boldsymbol{U}(\boldsymbol{X}) $ |$\frac{\partial{g(\boldsymbol{U})}}{\partial{\boldsymbol{X}}_{ij}} $ | $\text{tr}\left(\left(\frac{\partial{g(\boldsymbol{U})}}{\partial{\boldsymbol{U}}}\right)^{\mathsf{T}}\frac{\partial{\boldsymbol{U}}}{\partial{\boldsymbol{X}}_{ij}}\right) $ |

we here will give three examples of computations in detail.
### Example 1
To begin with, we will take the derivative of loss function $ L $ with respect to vector $ \boldsymbol{o} $. Using [denominator-layout](https://en.wikipedia.org/wiki/Matrix_calculus#Denominator-layout_notation) and chain rule, we have:
$$
\begin{aligned}
\nabla_{\boldsymbol{o}^{(t)}}L &= \left(
\frac{\partial{\boldsymbol{\hat{y}}}}{\partial{\boldsymbol{o}^{(t)}}}\frac{\partial{L}}{\partial{\boldsymbol{\hat{y}}}}
\right)
\end{aligned}  \tag{3.1.1}
$$

Taking the former derivative of $ \boldsymbol{\hat{y}} $ with respect to $ \boldsymbol{o} $, we get:
$$
\begin{aligned}
\frac
{\partial{\boldsymbol{\hat{y}}}}
{\partial{\boldsymbol{o}^{(t)}}}
&= 
\frac
{\partial{\frac{\text{exp}(\boldsymbol{o^{(t)}})}{\boldsymbol{1_c}^{\mathsf{T}}\text{exp}(\boldsymbol{o}^{(t)})}}}
{\partial{\boldsymbol{o}^{(t)}}}\\
&= 
\frac
{1}
{\boldsymbol{1_c}^{\mathsf{T}}\text{exp}(\boldsymbol{o}^{(t)})}
\frac
{\partial{\text{exp}(\boldsymbol{o}^{(t)})}}
{\partial{\boldsymbol{o}^{(t)}}}
+
\frac
{\partial{\frac
            {1}
            {\boldsymbol{1_c}^{\mathsf{T}}\text{exp}(\boldsymbol{o}^{(t)})}}}
{\partial{\boldsymbol{o}^{(t)}}}
\text{exp}(\boldsymbol{o}^{(t)})^{\mathsf{T}}\\
&=
\frac
{1}
{\boldsymbol{1_c}^{\mathsf{T}}\text{exp}(\boldsymbol{o}^{(t)})}
\text{diag}\left(
    \text{exp}(\boldsymbol{o}^{(t)})
    \right)
-\left({\frac
    {1}
    {\boldsymbol{1_c}^{\mathsf{T}}\text{exp}(\boldsymbol{o}^{(t)})}}
\right)^2
\text{exp}(\boldsymbol{o}^{(t)})
\text{exp}(\boldsymbol{o}^{(t)})^{\mathsf{T}}\\
&=
\text{diag}(\boldsymbol{\hat{y}}) - \boldsymbol{\hat{y}}{\boldsymbol{\hat{y}}}^{\mathsf{T}} 
\end{aligned} \tag{3.1.2}
$$

For the latter partial derivative, we have:
$$
\begin{aligned}
\frac
{\partial{L}}
{\partial{\boldsymbol{\hat{y}}}}
&=
\frac
{\partial{-\boldsymbol{y}^{\mathsf{T}}\text{log}(\boldsymbol{\hat{y}})}}
{\partial{\boldsymbol{\hat{y}}}}\\
&=
\frac{\partial{\text{log}(\boldsymbol{\hat{y}})}}{\partial{\boldsymbol{\hat{y}}}}
\frac{\partial{-\boldsymbol{y}^{\mathsf{T}}\text{log}(\boldsymbol{\hat{y}})}}{\partial{\text{log}(\boldsymbol{\hat{y}})}}\\
&=
\text{diag}(\boldsymbol{\hat{y}})^{-1}(-\boldsymbol{y})
\end{aligned} \tag{3.1.3}
$$

Therefore, gradient $ \nabla_{\boldsymbol{o}^{(t)}}L $ at time step $ t $ is as follows:
$$
\begin{aligned}
\nabla_{\boldsymbol{o}^{(t)}}L &= \frac{\partial{\boldsymbol{\hat{y}}}}{\partial{\boldsymbol{o}^{(t)}}}\frac{\partial{L}}{\partial{\boldsymbol{\hat{y}}}}\\
&=    
\left(
    \text{diag}(\boldsymbol{\hat{y}}) - \boldsymbol{\hat{y}}\boldsymbol{\hat{y}}^{\mathsf{T}}
    \right)
\left(
    \text{diag}(\boldsymbol{\hat{y}})^{-1}(-\boldsymbol{y})
    \right)\\
&=
\left(
\boldsymbol{I-\hat{y}}\boldsymbol{1_c}^{\mathsf{T}}
\right)(-\boldsymbol{y})\\
&=
\boldsymbol{\hat{y}-y}
\end{aligned} \tag{3.1.4}
$$
where the training target $ \boldsymbol{y} $ is a basic vector [0,...,0,1,0,...,0] with a 1 at the postion $ i $.

### Example 2
&emsp;Now, we consider a slightly more complicate example: computation of $\nabla_{\boldsymbol{h}^{(t)}}L$. 
Walking through computational graph backforward, it's easy to get the gradient at the final time step $ \tau $ because the $\boldsymbol{h}^{(t)}$ has only a descendent $\boldsymbol{o}^{(t)}$: 

$$
\nabla_{\boldsymbol{h}^{(t)}}L = \boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L \tag{3.2.1}
$$

However, we should note that the $\boldsymbol{h}^{(t)}$ has both $\boldsymbol{o}^{(t)}$ and $\boldsymbol{h}^{(t+1)}$ as descendents from $ t = \tau - 1 $ down to $t = 1$. Consequently, the gradient is given by:

$$
\nabla_{\boldsymbol{h}^{(t)}}L = 
\left(
    \frac{\partial{\boldsymbol{h}^{(t+1)}}}{\partial{\boldsymbol{h}^{(t)}}}
\right)^{\mathsf{T}}\nabla_{\boldsymbol{h}^{(t+1)}}L
+
\boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L \tag{3.2.2}
$$

Taking the two cases above into account, we have:
$$
\begin{aligned}
\nabla_{\boldsymbol{h}^{(t)}}L 
&= 
\begin{cases}
\boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L, & \qquad \text{if t at the final time step }\tau \\
\boldsymbol{W}^{\mathsf{T}}\text{diag}\left(
    1-(\boldsymbol{h}^{(t+1)})^2
    \right)
    (\nabla_{\boldsymbol{h}^{(t+1)}}L)
+
\boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L, & \qquad 1\le t \le \tau - 1\\
\end{cases}\\
\end{aligned} \tag{3.2.3}
$$

### Example 3
&emsp;Once the gradients on internal nodes were computed, we can take the derivatives of the parameters, e.g. $ \boldsymbol{W} $, $ \boldsymbol{V} $. Notice that the parameters are shared across the time steps, we will introduce some dummy variables such as $\boldsymbol{W}^{(t)}$ that are define to be copies of parameter but with each only used at time step $t$, which will be accumulated from $ t = \tau $ down to $ t = 0 $ so that we can obtain the gradient $ \nabla_{\boldsymbol{W}}L $ at the end of a backward propagation. 
Applying the notations above, the gradient on the parameter $\nabla_{\boldsymbol{V}}L $ is given by:

$$
\begin{aligned}
\nabla_{\boldsymbol{V}}L &= \sum_{t = 0}^{\tau}\nabla_{\boldsymbol{V}^{(t)}}L\\
&= 
\sum_{t = 0}^{\tau}\left[
\begin{matrix}
\text{tr}\left(
(\nabla_{\boldsymbol{o}^{(t)}}L)^{\mathsf{T}}\frac{\partial{\boldsymbol{o}^{(t)}}}{\partial{\boldsymbol{V}_{11}}}
\right) &\cdots & \text{tr}\left(
(\nabla_{\boldsymbol{o}^{(t)}}L)^{\mathsf{T}}\frac{\partial{\boldsymbol{o}^{(t)}}}{\partial{\boldsymbol{V}_{1n}}}
\right)\\ 
\vdots &\ddots & \vdots\\
\text{tr}\left(
(\nabla_{\boldsymbol{o}^{(t)}}L)^{\mathsf{T}}\frac{\partial{\boldsymbol{o}^{(t)}}}{\partial{\boldsymbol{V}_{n1}}}
\right) &\cdots & \text{tr}\left(
(\nabla_{\boldsymbol{o}^{(t)}}L)^{\mathsf{T}}\frac{\partial{\boldsymbol{o}^{(t)}}}{\partial{\boldsymbol{V}_{nn}}}
\right) 
\end{matrix}\right]
\\
&=
\sum_{t = 0}^{\tau}\left[\begin{matrix}
(\nabla_{\boldsymbol{o}^{(t)}}L)_{1}\boldsymbol{h}^{(t)}_{1} & \cdots & (\nabla_{\boldsymbol{o}^{(t)}}L)_{1}\boldsymbol{h}^{(t)}_{n}\\
\vdots &\ddots & \vdots\\
(\nabla_{\boldsymbol{o}^{(t)}}L)_{n}\boldsymbol{h}^{(t)}_{1} & \cdots & (\nabla_{\boldsymbol{o}^{(t)}}L)_{n}\boldsymbol{h}^{(t)}_{n}
\end{matrix}\right]
\\
&=
\sum_{t = 0}^{\tau}(\nabla_{\boldsymbol{o}^{(t)}}L){\boldsymbol{h}^{(t)}}^{\mathsf{T}}
\end{aligned} \tag{3.3.1}
$$

In fact, inspired by the relationship between total derivative and partial derivative in multivariable calculus:
$$
\text{d}f = \sum_{i=1}^{n}\frac{\partial{f}}{\partial{x_i}}\text{d}x_i \tag{3.3.2}
$$
which can be written in vector form as:
$$
\text{d}f = \left(
\frac{\partial{f}}{\partial{\boldsymbol{x}}}
\right)^{\mathsf{T}}
\text{d}\boldsymbol{x} \tag{3.3.3}
$$
we can obtian the relationship between total derivative and partial derivative in matrix calcalus:
$$
\begin{aligned}
\text{d}f &= \sum_{i=1}^{m}\sum_{j=1}^{n}
\frac{\partial{f}}{\partial{\boldsymbol{X}_{ij}}}\text{d}\boldsymbol{X}_{ij}\\
&=
\text{tr}\left(
\left(
    \frac{\partial{f}}{\partial{\boldsymbol{X}}}
\right)^{\mathsf{T}}
\text{d}\boldsymbol{X}
\right)
\end{aligned} \tag{3.3.4}
$$
Now, we can skip the tedious process (3.3.1) by applying eq. (3.3.4) for gradient $ \nabla_{\boldsymbol{V}}L $:
$$
\begin{aligned}
\text{d}L &= \text{tr}\left(
\left(
\frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\text{d}\boldsymbol{o}^{(t)}
\right)\\
&=
\text{tr}\left(
\left(
    \frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\text{d}\left(
\boldsymbol{c} + \boldsymbol{V}^{(t)}\boldsymbol{h}^{(t)}
\right)
\right)\\
&=
\text{tr}\left(
\left(
    \frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\left(
\left(\text{d}
\boldsymbol{V}^{(t)}
\right)\boldsymbol{h}^{(t)}
+
\boldsymbol{V}^{(t)}\text{d}\boldsymbol{h}^{(t)}
\right)
\right)\\
&=
\text{tr}\left(
\left(
    \frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\left(\text{d}
\boldsymbol{V}^{(t)}
\right)\boldsymbol{h}^{(t)}
\right)\\
&=
\text{tr}\left(
    \boldsymbol{h}^{(t)}
\left(
    \frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\left(\text{d}
\boldsymbol{V}^{(t)}
\right)
\right) 
\end{aligned} \tag{3.3.5}
$$

We could derive the result by comparing eq. (3.3.4) with eq. (3.3.5):
$$
\begin{aligned}
\nabla_{\boldsymbol{V}^{(t)}}L &= \frac{\partial{L}}{\partial{\boldsymbol{V}^{(t)}}}\\
&=
\left(
\boldsymbol{h}^{(t)}
\left(
    \frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}
\right)^{\mathsf{T}}
\right)^{\mathsf{T}}\\
&=
\frac{\partial{L}}{\partial{\boldsymbol{o}^{(t)}}}{\boldsymbol{h}^{(t)}}^{\mathsf{T}}\\
&=
\left(
\nabla_{\boldsymbol{o}^{(t)}}L
\right){\boldsymbol{h}^{(t)}}^{\mathsf{T}}
\end{aligned}   \tag{3.3.6}
$$

$$
\begin{aligned}
\nabla_{\boldsymbol{V}}L &= \sum_{i=0}^{\tau}\nabla_{\boldsymbol{V}^{(t)}}L\\
&=
\sum_{i=0}^{\tau}
\left(
\nabla_{\boldsymbol{o}^{(t)}}L
\right){\boldsymbol{h}^{(t)}}^{\mathsf{T}}
\end{aligned}   \tag{3.3.7}
$$

Similarly, using the equation (3.3.4), the gradient on remaining parameters is quite easy to get and we can gather these gradients as follows:
$$
\begin{align*}
\nabla_{\boldsymbol{o}^{(t)}}L &= \boldsymbol{\hat{y}-y} \tag{3.3.8} \\
\nabla_{\boldsymbol{h}^{(t)}}L 
&= 
\begin{cases}
\boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L, & \qquad \text{if t at the final time step }\tau \\
\boldsymbol{W}^{\mathsf{T}}\text{diag}\left(
    1-(\boldsymbol{h}^{(t+1)})^2
    \right)
    (\nabla_{\boldsymbol{h}^{(t+1)}}L)
+
\boldsymbol{V}^{\mathsf{T}}\nabla_{\boldsymbol{o}^{(t)}}L, & \qquad 1\le t \le \tau - 1\\
\end{cases} \tag{3.3.9} \\
\nabla_{\boldsymbol{c}}L &= \sum_{t}\nabla_{\boldsymbol{o}^{(t)}}L \tag{3.3.10} \\
\nabla_{\boldsymbol{b}}L &= \sum_{t}\text{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right)\nabla_{\boldsymbol{h}^{(t)}}L \tag{3.3.11} \\
\nabla_{\boldsymbol{V}}L &= \sum_{t}\left(
\nabla_{\boldsymbol{o}^{(t)}}L
\right){\boldsymbol{h}^{(t)}}^{\mathsf{T}} \tag{3.3.12} \\
\nabla_{\boldsymbol{W}}L &= \sum_{t}\text{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right)\left(\nabla_{\boldsymbol{h}^{(t)}}L\right){\boldsymbol{h}^{(t-1)}}^{\mathsf{T}} \tag{3.3.13} \\
\nabla_{\boldsymbol{U}}L &= \sum_{t}\text{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right)\left(\nabla_{\boldsymbol{h}^{(t)}}L\right){\boldsymbol{x}^{(t)}}^{\mathsf{T}} \tag{3.3.14}
\end{align*}
$$
