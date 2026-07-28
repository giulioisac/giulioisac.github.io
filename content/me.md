---
title: Maximum Entropy, Contrastive Learning and Logistic Regression
description: How the principle of maximum entropy connects unsupervised learning and density-ratio estimation, and its link to logistic regression.
tag: Notes / ML Theory
blurb: How the principle of maximum entropy connects unsupervised learning and density-ratio estimation, and its surprising link to logistic regression.
order: 1
math: true
script: assets/figs.js
---

The [principle of maximum entropy](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy), first introduced by Jaynes in the 60' and inspired by similarities between information theory and statistical mechanics, is a powerful and popular method for modeling real-world data with minimal assumptions. In this post I will discuss its connection to unsupervised learning and density ratio estimation. This connection will suggest a surprising relationship to the most popular inference scheme for supervised learning: logistic regression.

## Unsupervised Learning

Unsupervised learning addresses the task of inference of a probability distribution \(P(x)\) from a set of samples \(\mathcal{D} = \{ x_i \}^N_{i=1}\). Given a specific parametrization of the probability distribution \(P^\theta(x)\), parameter inference is performed via maximization of the average log-likelihood

\[ \mathcal{L}(\theta,\mathcal{D})=\frac{1}{N}\sum_{i=1}^N \log P^\theta(x_i)=\mathbb{E}_\mathcal{D}[\log P^\theta] \]

where \(\mathbb{E}_\mathcal{D}[\cdot]\) indicates the empirical average with respect to the set \(\mathcal{D}\). We have a method for parameter inference; we need now to find a heuristic to decide the functional form of \(P(x)\), this is when maximum entropy and the principle of minimum discrimination information can help us.

## Principle of Minimum Discrimination Information

We are looking for a probability distribution \(P(x)\) that is the most similar to a reference distribution \(P_{\rm 0}\) while reproducing some average observables from the data \(\mathcal{D}\) (such as the mean and standard deviation of \(x\))

\[ \mathcal{J}(P)=\mathbb{D}_{KL}(P||P_{\rm 0}) - \eta_0 \left(\sum_x P(x) -1 \right) + \sum_{f \in \mathcal{F}} \theta_f \left( \mathbb{E}_P[g_f] - \mathbb{E}_{P_{\rm data}}[g_f] \right) \]

where the second term imposes normalization of the probability, the index \(f\) identifies the observables \(g_f(x)\) whose average is constrained to match the one computed on samples \(\mathcal{D}\) from the \(P_{\rm data}\) distribution and

\[ \mathbb{D}_{KL}(P||P_{\rm 0})= \sum_x P(x) \log \left(\frac{P(x)}{P_{\rm 0}(x)}\right) \]

is the Kullback-Leibler divergence between the two distributions. The uniform distribution \(\mathcal{U}\) is the distribution with maximal entropy. When \(P_{\rm 0}\to \mathcal{U}\) the above formulation is equivalent to the principle of maximum entropy. Extremization of \(\mathcal{J}\) leads to

\[ P^\theta(x) = \frac{1}{Z^\theta}e^{-E^\theta(x)} P_{\rm 0 }(x) \]

with \(E^\theta(x)=\sum_{f}\theta_f g_f(x)\) where the function \(g_f\) defines the observables that we want to match, i.e. \(g(x)=x\) and \(g(x)=x^2\) for the first and second moments respectively. The partition function \(Z^\theta\) can be estimated through importance sampling as

\[ Z^\theta=\sum_x P_{\rm 0}(x)e^{-E^\theta(x)}\sim \mathbb{E}_\mathcal{G} [e^{-E^\theta}] \]

with \(\mathbb{E}_\mathcal{G}[\cdot]\) the empirical average with respect to a set \(\mathcal{G}\) of generated samples from \(P_{\rm 0}\). Intuitively, when \(P_{\rm 0}(x)\) is closer to \(P(x)\), the empirical average for \(Z^\theta\) converges faster and the inference is more stable.

Figure 1 (top) shows the tilt at work: a broad reference \(P_{\rm 0}\) is reweighted by the Boltzmann factor \(e^{-E(x)}\) into the distribution \(P\) that matches the constrained moments of the data. The bottom panel belongs to the next section, and we come back to it once the classifier has been introduced.

{{figure: me-figClf}}

## Connection to Logistic Regression

Parameter inference of \(E^\theta(x)\) is commonly difficult because the estimation of \(Z^\theta\) is computationally intensive. Luckily, we can get around this problem via optimization of an alternative objective function, which shares the same maximum: the [binary cross entropy loss](https://en.wikipedia.org/wiki/Logistic_regression).

In the [Noise Contrastive Estimation](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) framework, which developed within the field of Natural Language Processing, the density ratio between two distributions \(P(x)/P_{\rm 0}(x)\) is inferred using a logistic classifier. A classifier between the two hypotheses of \(x\) originating from \(P(x)\) or \(P_{\rm 0}\) in a mixture \(P_{\rm mix}(x)=\frac{1}{2}(P(x)+ P_{\rm 0}(x))\) follows:

\[ d(x)=\frac{P(x)}{P(x)+P_{\rm 0}(x)}=\frac{1}{1+Ze^{E(x)}} \]

if we parametrize the classifier \(d=d^\theta\), we can minimize the binary cross-entropy

\[ S(\theta)=-\mathbb{E}_\mathcal{D}[\log d^\theta]-\mathbb{E}_\mathcal{G}[\log (1-d^\theta)] \]

and recover the density ratio using the energy representation of \(d^\theta\). This is the bottom panel of Figure 1: the optimal classifier crosses \(d=\tfrac{1}{2}\) exactly where the two densities intersect, and its logit is the log density ratio \(-E(x)-\log Z\), so reading the classifier off recovers the energy. Note that \(Z^\theta\) never needs to be computed during training: it enters the logit of \(d^\theta\) only as an additive constant \(-\log Z^\theta\), which can be learned as one extra parameter, the intercept of the classifier, together with \(\theta\) (this is exactly how the demo in Figure 2 treats it). Logistic regression (classification) is the workhorse of supervised learning. It finds here an untraditional application to an unsupervised problem. It is also the seed of much of modern contrastive learning: the discriminator of a GAN estimates exactly this \(d(x)\), and negative-sampling objectives such as word2vec's and InfoNCE descend directly from the NCE loss.

Figure 2 runs the whole scheme in the browser. Samples are drawn from an unknown bimodal \(P(x)\) and from a Gaussian noise distribution \(P_{\rm 0}(x)\); a logistic classifier over a set of basis functions is fitted by gradient descent, and the density it implies, \(e^{-E^\theta(x)}P_{\rm 0}(x)/Z^\theta\), is drawn as it converges. Move the noise distribution away from the data and watch the effective sample size behind the importance-sampling estimate of \(Z^\theta\) collapse: the practical face of the remark above that inference is more stable when \(P_{\rm 0}\) is close to \(P\).

{{figure: me-figDemo}}

## Why do we care? The road to explainable blackboxes

While the Maximum Entropy description allows us to minimize the amount of structure we build in the model, sometimes we want to impose additional structure. One example is the estimation of [selection pressure](https://www.pnas.org/doi/abs/10.1073/pnas.2023141118), another one is [simulation based inference](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.055309). In another post we also discussed [model calibration](bdt.html).

Inference via density ratio estimation is a powerful method to develop explainable models. When \(P_0(x)\) represents your simple model and the energy \(E^\theta(x)\) is parametrized via a deep neural network, we can bridge the predictions of \(P_0(x)\) to the complexity of real-world data and its varied sources of noise.
