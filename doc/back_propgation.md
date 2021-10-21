Grad for $W_3$, shape($W_3$) = $L_2 * L_3$:
$W = [W_1,W_2,W_3]\\$
We define loss function: $L(x;W)= [\sigma(W_3*\alpha_2) - y]^2\\$
Then gradients for $W_3$ can be drive as:$\\$
$\frac{dL(x;W)}{dW_3} =\frac{dL(x;W)}{d\alpha_3} * \frac{d\alpha_3}{dW_3} = 2 * [\sigma(W_3*\alpha_2) - y] * \frac{d\sigma(W_3*\alpha_2)}{dW_3} \\$


Some hint about $\sigma(x):\\$
$\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^{x}}{1+e^{x}}\\$
$\sigma(-x) = \frac{1}{1+e^{x}} = 1 - \frac{e^x}{1+e^x} = 1 - \sigma(x)\\$
$\frac{d\sigma(x)}{dx} = \frac{e^x*(1+e^x) - e^x*e^x}{(1+e^x)^2} = \frac{e^x}{(1+e^x)^2} = \sigma(x) * \sigma(-x) =\sigma(x) * (1-\sigma(x))\\$


Then we can get $\\$
$\frac{d\sigma(W_3*\alpha_2)}{dW_3} = \sigma(W_3*\alpha_2) * (1 - \sigma(W_3*\alpha_2)) * \alpha_2$

Some hint about chain rule: $\frac{df(g(x))}{dx} = \frac{df}{dg}*\frac{dg}{dx}$

For layer 2 $\\$
$\frac{dL(x;W)}{dW_2} = \frac{dL(x;W)}{d\alpha_3} * \frac{d\alpha_3}{d\alpha_2} * \frac{d\alpha_2}{dW_2}$

Foe layer 1 $\\$
$\frac{dL(x;W)}{dW_1} = \frac{dL(x;W)}{d\alpha_3} * \frac{d\alpha_3}{d\alpha_2} * \frac{d\alpha_2}{d\alpha_1} * \frac{d\alpha_1}{dW_1}$


For N layer MLP:
$\frac{dL(x;W)}{dW_1} = \frac{dL(x;W)}{d\alpha_N} * \frac{d\alpha_N}{d\alpha_{N-1}} *...* \frac{d\alpha_2}{d\alpha_1} * \frac{d\alpha_1}{dW_1}$

Shape of $W_N$ = $L_{N-1} * L_N$, $\alpha_N$ = $L_N$,$\alpha_{N-1}$ = $L_{N-1}$ what's the shape of $\frac{d\alpha_N}{d\alpha_{N-1}}$ ?

Shape of $x = M , W_1 = M * L_1$

Shape of $\frac{dL(x;W)}{dW_1} = \frac{dL(x;W)}{d\alpha_N} * \frac{d\alpha_N}{d\alpha_{N-1}} *...* \frac{d\alpha_2}{d\alpha_1} * \frac{d\alpha_1}{dW_1} 
\\$

$[M * L_1] = [1,L_N] * [L_N,L_{N-1}] ... [L_2,L_1] * [L_1,M,L_1]
\\$

Suppose each layer have same units $L$, then:

$[M * L] = [1,L] * [L,L] ... [L,L] * [L,M,L]
\\$
 
 $G_{M * L} = [1,L] * \prod_{i=n-1} L_i * [L,M,L]
\\$

 $L_i = P_i * [\lambda_1^{i},\lambda_2^{i},...,\lambda_L^{i}]*P_i^{-1} = P_i * \lambda_{i}* P_i^{-1}
\\$

Suppose $L_i \approx L_j $(or let L be i.i.d)

$L_{N-1} * L_{N-2} = P_{N-1} * \lambda_{N-1} * P_{N-1}^{-1} * P_{N-2} * \lambda_{N-2} * P_{N-2}^{-1} \\$
$\approx P_{N-1} * \lambda^2 * P_{N-2}^{-1}
\\$

$\prod_{i=n-1} L_i = P_{N-1} * \lambda_{N-1} * P_{N-1}^{-1} * ... * P_1 * \lambda_{1}* P_1^{-1}\\$
$
\approx P_{N-1} * \lambda^{N-1} * P_{1}^{-1}
\\$

What is $\lambda^{N-1}$ ?

$ \lambda^{N-1} = [\lambda_1^{N-1},\lambda_2^{N-1},...,\lambda_L^{N-1}]
\\$

if $|\lambda_i| < 1$, then $\lambda_i^{N-1}$ converge to 0 which lead $G$ to vanish. 

if $|\lambda_i| > 1$, then $\lambda_i^{N-1}$ converge to $\infty$, which lead $G$ to explosion.

# Residual

<!-- ### DNN layer $N-1$
$\frac{dL(x;W)}{dW_N} =\frac{dL(x;W)}{d\alpha_{N}} * \frac{d\alpha_{N}}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{dW_{N-1}}
$ -->

### DNN layer $N-2$
$\frac{dL(x;W)}{dW_{N-2}} =\frac{dL(x;W)}{d\alpha_{N}} * \frac{d\alpha_{N}}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{d\alpha_{N-2}} * \frac{d\alpha_{N-2}}{dW_{N-2}}
$

<!-- ### Resnet layer $N-1$ :
$\frac{dL(x;W)}{dW_N} =\frac{dL(x;W)}{d\alpha_N+s_T} * \frac{d\alpha_N+s_T}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{dW_{N-1}}
$ -->

### Resnet layer $N-2$ :
$\frac{dL(x;W)}{dW_{N-2}} =\frac{dL(x;W)}{d\alpha_N+s_T} * \frac{d\alpha_N+s_T}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{d\alpha_{N-2}} * \frac{d\alpha_{N-2}}{W_{N-2}}
$

<!-- ### Resnet layer $N-3$ :
$\frac{dL(x;W)}{dW_N} =\frac{dL(x;W)}{d\alpha_N+s_T} * \frac{d\alpha_N+s_T}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{d\alpha_{N-2}} * \frac{d\alpha_{N-2}}{d\alpha_{N-2}+s_{T-1}} *\frac{d\alpha_{N-2}+s_{T-1}}{d\alpha_{N-3}} *\frac{d\alpha_{N-3}}{W_{N-3}} 
$-->
...

### $s_T = \alpha_{N-2}$ , then $T = N-2$.

For resnet layer $N-2$, we notice that,
<!-- $\frac{dL(x;W)}{dW_N} =\frac{dL(x;W)}{d\alpha_N+\alpha_{N-2}} * \frac{d\alpha_N+\alpha_{N-2}}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{d\alpha_{N-2}} * \frac{d\alpha_{N-2}}{W_{N-2}}
$ -->

$\frac{d\alpha_N+\alpha_{N-2}}{d\alpha_{N-1}} * \frac{d\alpha_{N-1}}{d\alpha_{N-2}} = [\frac{d\alpha_N}{d\alpha_{N-1}}  + \frac{d\alpha_{N-2}}{d\alpha_{N-1}}]* \frac{d\alpha_{N-1}}{d\alpha_{N-2}} = \frac{d\alpha_N}{d\alpha_{N-1}}*\frac{d\alpha_{N-1}}{d\alpha_{N-2}} + E
$

Then,
$\frac{dL(x;W)}{dW_{N-2}} =\frac{dL(x;W)}{d\alpha_N+\alpha_{N-2}} * [\frac{d\alpha_N}{d\alpha_{N-1}}*\frac{d\alpha_{N-1}}{d\alpha_{N-2}} + E] * \frac{d\alpha_{N-2}}{W_{N-2}}
$

For layer 1:
$\frac{dL(x;W)}{dW_{1}} =\frac{dL(x;W)}{d\alpha_N+s_T} * [\frac{d\alpha_N}{d\alpha_{N-1}}*\frac{d\alpha_{N-1}}{d\alpha_{N-2}} + E] * [\frac{d\alpha_{N-2}}{d\alpha_{N-3}}*\frac{d\alpha_{N-3}}{d\alpha_{N-4}} + E]*...*\frac{d\alpha_1}{dW_1}
$

Firstly, $| \lambda | >= 1$, which prevent from gradient vanish.

Secondly, if two layer is exemely similar where $\alpha_N \approx \alpha_{N-1}$, which means $\frac{d\alpha_N}{d\alpha_{N-1}} + E \approx E$ and results in the behavior like reduced total nums of neighbor layers with similar output.