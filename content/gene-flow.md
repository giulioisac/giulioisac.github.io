---
title: Migration Networks from Neutral Allele Frequencies
description: Why neutrality forces a linear model of gene flow, and how one model reads transmission networks from SARS-CoV-2 and migration networks from ancient DNA.
tag: Notes / Population Genetics
blurb: One linear model of neutral allele frequencies reads transmission networks from SARS-CoV-2 and ancient migrations from archaeogenetic data.
order: 3
math: true
unlisted: true
script: assets/gene-flow.js
---

During the Delta wave, genomic surveillance in England was producing tens of thousands of SARS-CoV-2 genomes a week. Follow a common mutation in two regions, tracking the fraction of genomes in each that carry it: while travel between them is suppressed the two frequencies tend to wander apart, and once travel resumes they converge again. Nobody recorded the journeys that caused the convergence, yet their imprint was left for us to read.

Whether that imprint is there at all depends on which mutations we choose to follow, and the ones tracked here are the ones that do not matter. A neutral mutation is neither favored nor purged, so on average the only thing that moves its frequency in a region is the arrival and departure of the individuals carrying it. Left alone it still wanders — that is genetic drift, and it is the noise the rest of this note has to survive — but it wanders with no preferred direction, and an expectation is a thing an estimator can be built on. Migration is the only thing left pushing in a direction, and that is what makes a neutral mutation readable. A selected mutation instead has a direction of its own: its frequency is expected to move in a closed population as well, and once a rise can be produced from within there is no way to tell it from a rise produced by arrivals. What follows is the single linear equation that turns this convergence into an estimate of who moved where, and its application to two datasets with almost nothing in common beyond that neutrality: weekly SARS-CoV-2 genomes across regions, and ancient human genomes sampled centuries apart.

## A Metapopulation of Neutral Alleles

We consider a population divided into \(n\) subpopulations (regions, districts, age groups, etc) and follow the frequency \(X_i(t)\) of one neutral allele in each. Under neutrality the frequencies at a later time depend linearly on the frequencies at an earlier one:

\[ X_i(t+\Delta t) = \sum_{j=1}^n A_{ij}\, X_j(t) + \eta_i \]

where \(A\) is a right-stochastic matrix, its elements non-negative and summing to one within each row, and \(\eta_i\) is genetic drift, about which the least-squares estimators below need nothing except that its expectation vanishes. The deterministic term is linear because under neutrality the frequency of any union of lineages must obey the same evolution equation, which makes the map additive over disjoint groups of lineages, and because the conditional expectation of a neutral frequency is extensive: starting with half as many mutants is expected to lead to half as many mutants later. Additive and extensive leaves nothing but a linear map. Neutrality also fixes \(A\) to be the same matrix for every allele we track, since it describes who moved and not what they carried, and that is what allows many loci to be pooled into one estimate. The coefficient \(A_{ij}\) is the proportion of individuals in population \(i\) that originated from population \(j\) during the interval, the fraction of \(i\) replaced by migrants from \(j\). Read epidemiologically it is the proportion of infections that \(i\) imports from \(j\), which is why the COVID work{{cite: okada2025}} calls \(A\) the importation-rate matrix and the ancient DNA work{{cite: isacchini2026}} the migration matrix.

The same matrix can be read in the opposite direction of time, and that reading is worth having. Take a genome sampled in population \(i\) and follow its lineage of ancestors into the past. Each row of \(A\) sums to one over non-negative entries, which makes it a probability distribution over source populations, so \(A_{ij}\) is the probability that the lineage sat in \(j\) one interval earlier and \(A_{ii}\) the probability that it stayed put. Forward in time \(A\) is a table of migrant proportions; backward in time it is the transition matrix of a Markov chain on demes, and the history of a single lineage is one walk of that chain.

If we subtract \(X_i(t)\) from both sides, the mechanism is in plain view:

\[ X_i(t+\Delta t) - X_i(t) = \sum_{j=1}^n A_{ij}\,\bigl[X_j(t) - X_i(t)\bigr] + \eta_i \]

where the bracket shows that \(j\) influences \(i\) only when the two frequencies differ, and that a larger \(A_{ij}\) makes \(X_i\) converge on \(X_j\) faster. Migration is a spring and drift is the noise that keeps it from ever settling.

## Two Impossible Datasets

The model is the easy part. What separates one application from the next is which statistic of the data survives the noise, and the two datasets we are comparing fail in opposite directions.

SARS-CoV-2 surveillance is dense in time and thin in loci. England was sequenced weekly across nine regions, so \(\Delta t\) is one week and there are dozens of consecutive time points, but a 30-kilobase genome under a strong clonal background offers only a few tens of approximately independent segregating alleles. Ancient DNA is the mirror image. The Allen Ancient DNA Resource{{cite: mallick2024}} genotypes individuals at up to 1.23 million positions, which is an enormous number of loci, and delivers them as a handful of skeletons per 300-year window, pseudohaploid, with the great majority of positions missing in any one sample and the dates themselves uncertain by decades to centuries.

## Frequencies We Can Trust

Suppose for a moment that the true frequencies are known. Then the estimator writes itself: minimizing the squared difference between predicted and observed frequencies over all right-stochastic matrices gives

\[ A^{\rm (LS)} = \operatorname*{arg\,min}_{A} \sum_{i,\,\mu,\,t} \Bigl[ X^\mu_i(t+\Delta t) - \sum_j A_{ij} X^\mu_j(t) \Bigr]^2 \]

where \(\mu\) labels the independent loci and the minimization runs subject to \(A_{ij}\ge 0\) and \(\sum_j A_{ij}=1\). Summing over a window of time points regularizes the fit and sets the resolution at which \(A\) is allowed to vary. This convex problem is the target in both applications, and the rest of this section and the next are the two ways of reaching it when the frequencies cannot be taken at face value.

The estimator wants the true frequencies and we only ever see a noisy sample of them. The remedy in the COVID work is to stop treating what we measured as the state of the system. A hidden Markov model keeps the true frequencies as hidden states and gives genetic drift and sampling their own noise terms, with drift variance set by an effective population size and sampling variance set by the number of sequences and a per-region overdispersion. Gaussian noise makes the model a Kalman filter with an analytic likelihood, and the parameters follow from MCMC or, more cheaply, from expectation-maximization. Figure 1 shows the structure of this model.

{{figure: gene-flow-figKalman}}

## Distances We Can Trust

None of that survives contact with ancient DNA. A few genomes in a 300-year bin do not determine a frequency, the noise at that depth is not remotely Gaussian, and each sample covers a different subset of sites. We give up on individual frequencies instead, and write the dynamics directly in terms of quantities that many loci can estimate together.

The \(F_2\) statistic{{cite: patterson2012}} is the simplest of them: the squared difference in allele frequency between two populations, averaged over the whole genome. No single site is measurable at this sample size, but the average over a million of them is, and because every site contributes its own unbiased term, samples covering different subsets of positions still combine into one number. Under neutrality its expectation grows with the drift that separates two populations, which is why the field reads it as a genetic distance and builds admixture graphs out of it.

Write \(F_{ik}=\langle (X_i-X_k)^2\rangle\) for that distance and \(F'_{ik}=\langle (X'_i-X_k)^2\rangle\) for its lagged version, where the prime denotes time \(t+\Delta t\) and the brackets are the average over sites. The lag is the addition here, and it is what tells a donor from a recipient.

Figure 2 runs both statistics in the browser for two demes, \(i\) and \(k\), the two sliders being the only off-diagonal entries \(A\) has when there are two of them. The top panel is the relaxation of the model itself, the neutral frequencies in \(i\) and \(k\) pulled together by migration, and the bottom panel is the distance between them with its two lagged versions, \(F'_{ik}\) and \(F'_{ki}\). Symmetric migration pulls all three curves down together. Set the sliders so that flow runs in one direction only and the two separate, because only the receiving population has moved toward where the donor used to be.

{{figure: gene-flow-figRelax}}

The derivation costs nothing. Split the lagged difference as \(X'_i - X_k = (X'_i - X_i) + (X_i - X_k)\), how far \(i\) moved plus how far apart the two populations were, then square it, average over loci, and substitute the relaxation form of the model{{cite: isacchini2026}}. The cross term between drift and the current difference has zero expectation, since drift does not know how far apart the populations already are, and what survives is a relation between distances alone:

\[ F'_{ik} - F'_{ii} = \sum_{j=1}^n A_{ij}\,\bigl(F_{jk} - F_{ji}\bigr) \]

where \(F'_{ii}\) is the mean squared displacement of population \(i\) over the interval, every quantity on both sides is a measurable genetic distance, and the dependence on \(A\) is still linear. Genetic drift has vanished from the expectation. Its variance lives in \(F'_{ii}\), which enters \(F'_{ik}\) identically and cancels in the difference \(F'_{ik} - F'_{ii}\).

Sampling noise leaves the same way, and this is what makes the statistic usable at a few genomes per bin. An independent error of variance \(v_a\) in the estimate for population \(a\) inflates every distance it touches, \(\hat F_{jk} = F_{jk} + v_j + v_k\), so the right-hand side picks up \(\sum_j A_{ij}(v_k - v_i) = v_k - v_i\) and the left-hand side picks up exactly the same \(v_k - v_i\), once again because the rows of \(A\) sum to one. The relation holds for the noisy estimates unchanged, with no correction to apply and nothing about the sampling to model. From there \(A\) is fit as it was before, by least squares on the residuals of this relation over all \(i\), \(k\) and \(t\), subject to the same non-negativity and row-sum constraints: the same convex program, run on distances instead of frequencies.

## Rates Instead of Proportions

The appeal of this construction is how little it assumes. There is no phylogeny to reconstruct and no ancestral source population to postulate, which matters because both are modeling choices that quietly determine the answer. What comes out is a matrix of rates over an interval, estimated by a convex program — wrapped in a filter only where the frequencies themselves have to be inferred — that scales to whatever sequencing produces next.

Each entry is still a proportion, as it was when we defined it, but a proportion per interval rather than a single number standing for all time, and that is what generalizes the quantity the field already measures. Multiplying the matrices over successive intervals, \(A(t)\,A(t-1)\cdots A(1)\), gives a matrix whose rows are the expected proportions of ancestry in each population traceable to each population at the starting time. That is precisely an admixture proportion, with the restriction to a single static episode lifted. It also says what one ancient-DNA interval is: a 300-year bin is already such a product, ten generations compounded into one step, to be read as net flow rather than a per-generation rate. Admixture proportions answer where ancestry came from. A product of stochastic matrices answers when it moved, and by which route.

## Wherever Something Is Sampled Twice

The two analyses are one model and two estimators, run at opposite ends of a single axis. SARS-CoV-2 gives many time points and few independent alleles, so the noise has to be modeled, and a Kalman filter pulls the true frequencies out from under drift and sampling. Ancient DNA gives a million loci and almost no time points, so the noise has to be cancelled, and the difference of two lagged distances does it without a noise model at all. What comes back is the same object either way. In England and the United States the networks track geography but carry long-range links stronger than mobility data predicts, and they change between variant waves{{cite: okada2025}}; in Europe the Steppe ancestry that reaches Britain in the third millennium turns out to arrive by way of Central Europe rather than directly, a route with a direction and a date where a static ancestry proportion shows only a component{{cite: isacchini2026}}.

Nothing in the derivation knows that the populations are places. It asks only for subpopulations, variants nobody is selecting on, and samples at more than one time. Communities split by age or socioeconomic status rather than by geography would serve as well{{cite: okada2025}}, as would any pathogen under routine surveillance, or the microbial populations that metagenomics reads repeatedly from the same hosts and sites, where dispersal between them does exactly what migration does here.

One limit travels with the method wherever it goes. It sees movement only through the divergence that movement destroys, so exchange between populations that already look alike leaves no imprint, however much of it there is. What it can read is set by the divergence the data happen to contain, not by the inference.

## References

1. {ref: okada2025} [Okada et al., *PNAS* **122**, e2500663122 (2025)](https://doi.org/10.1073/pnas.2500663122), the SARS-CoV-2 transmission networks, with a [commentary](https://doi.org/10.1073/pnas.2533093123) in the same journal.
1. {ref: isacchini2026} [Isacchini et al., *bioRxiv* (2026)](https://doi.org/10.64898/2026.03.12.710875), the ancient DNA analysis and the \(F_2\) formulation used here.
1. {ref: mallick2024} [Mallick et al., *Scientific Data* **11**, 182 (2024)](https://doi.org/10.1038/s41597-024-03031-7), the Allen Ancient DNA Resource, released through [Harvard Dataverse](https://doi.org/10.7910/DVN/FFIDCW).
1. {ref: patterson2012} [Patterson et al., *Genetics* **192**, 1065 (2012)](https://doi.org/10.1534/genetics.112.145037), the \(f\)-statistics.