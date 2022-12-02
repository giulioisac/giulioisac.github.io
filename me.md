# Maximum Entropy, Contrastive Learning and Logistic Regression

The [principle of maximum entropy](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy), first introduced by Jaynes in the 60' 
and inspired by similarities between information theory and statistical mechanics, 
is a powerful and popular method for modeling real-world data with minimal assumptions. 
In this post I will discuss its connection to unsupervised learning and density ratio estimation. 
This connection will suggest a suprising relationship to the most popular inference scheme for supervised learning: logistic regression.

## Unsupervised Learning
Unsupervised learning addresses the task of inference of a probability distribution $$P(x)$$ from a set of samples $$D = \{ x_i \}^N_{i=1}$$. 
Given a specific parametrization of the probability distribution $$P^\theta(x)$$, parameter inference is performed 
via maximization of the average log-likelihood

$$
\begin{aligned}
\mathcal{L}(\theta,\mathcal{D})=\frac{1}{N}\sum_{i=1}^N \log P^\theta(x_i)=\mathbb{E}_\mathcal{D}[\log P^\theta]
\end{aligned}
$$

where $$\mathbb{E}_\mathcal{D}[\cdot]$$ indicates the empirical average with respect to the set $$\mathcal{D}$$.  

## Principle of Minimum Discriminatory Information
We are looking for a probability distribution $$P(x)$$ that is the most similar to a reference distribution $$P_{\rm 0}$$
while reproducing some average observables from the data $$\mathcal{D}$$ (such as the mean and standard deviation of x)

$$
\begin{aligned}
\mathcal{J}(P)=\mathbb{D}_{KL}(P||P_{\rm 0}) - \eta_0 \left(\sum_x P(x) -1 \right) - \sum_{f \in \mathcal{F}} \theta_f \left( P(f) - P_{\rm data}(f) \right)
\end{aligned}
$$

where the second term imposes normalization of the probability, the index $$f$$ identifies the observables 
whose average is constrained to match the one computed on samples $$\mathcal{D}$$ from the $$P_{\rm data}$$ distribution and 

$$
\begin{aligned}
\mathbb{D}_{KL}(P||P_{\rm 0})= \sum_x P(x) \log \left(\frac{P(x)}{P_{\rm 0}(x)}\right)
\end{aligned}
$$

is the Kullback-Leibler divergence between the two distributions. 
The uniform distribution $$\mathcal{U}$$ is the distribution with maximal entropy. 
When $$P_{\rm 0}\to \mathcal{U}$$ the above formulation is equivalent to the principle of maximum entropy. 
Extremization of $$\mathcal{J}$$ leads to 

$$
\begin{aligned}
P^\theta(x) = \frac{1}{Z^\theta}e^{-E^\theta(x)} P_{\rm 0 }(x) 
\end{aligned}
$$

with $$E^\theta(x)=\sum_{f}\theta^fg_f(x)$$ where the function $$g_f$$ defines the observables that we want to match, i.e. $$g(x)=x$$ and $$(g(x)=x^2)$$ for the first and second moments respectively.
The partition function $$Z^\theta$$ can be estimated through importance sampling as

$$
\begin{equation}
Z^\theta=\sum_x P_{\rm 0}(x)e^{-E^\theta(x)}\sim \mathbb{E}_\mathcal{G} [e^{-E^\theta}]
\end{equation}
$$

with $$\mathbb{E}_\mathcal{G}[\cdot]$$ the empirical average with respect to a set $$\mathcal{G}$$ of generated samples from $$P_{\rm 0}$$. 
Intuitively when $$P_{\rm 0}(x)$$ is closer to $$P(x)$$, the empirical average for $$Z^\theta$$ converges faster and the inference is more stable.


# Connection to Logistic Regression
Parameter inference of $$E^\theta(x)$$ is commonly difficult because the estimation of $$Z^\theta$$ is computationally intensive.
Luckly, we can get around this problem via optimization of an alternative objective function, which shares the same maximum: the [binary cross entropy loss](https://en.wikipedia.org/wiki/Logistic_regression).

In the [Noise Contrastive Estimation](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) framework, which developed within the field of Natural Language Processing, 
the density ratio between two distributions $$P(x)/P_{\rm 0}(x)$$ is inferred using a logistic classifier. 
A classifier between the two hypothesis of $$x$$ originating from $$P(x)$$ or $$P_{\rm 0}$$ in a mixture $$P_{\rm mix}(x)=\frac{1}{2}(P(x)+ P_{\rm 0}(x))$$  
follows:

$$
\begin{equation}
d(x)=\frac{P(x)}{P(x)+P_{\rm 0}(x)}=\frac{1}{1+Ze^{E(x)}}
\end{equation}
$$

if we parametrize the classifier $$d=d^\theta$$, we can minimize the binary cross-entropy

$$
\begin{equation}
S(\theta)=-\mathbb{E}_\mathcal{D}[\log d^\theta]-\mathbb{E}_\mathcal{G}[\log (1-d^\theta)]
\end{equation}
$$

and recover the density ratio using the energy representation of $$d^\theta$$.  
Logistic regression (classification) is the workhorse of supervised learning. 
It finds here an untraditional application to an unsupervised problem. 
