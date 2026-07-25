---
title: Migration Networks from Neutral Allele Frequencies
description: Why neutrality forces a linear model of gene flow, and how one equation reads transmission networks from SARS-CoV-2 and migration networks from ancient DNA.
tag: Notes / Population Genetics
blurb: One linear model of neutral allele frequencies reads transmission networks from SARS-CoV-2 and ancient migrations from archaeogenetic data.
order: 3
math: true
unlisted: true
script: assets/gene-flow.js
---

During the Delta wave, genomic surveillance in England was producing tens of thousands of SARS-CoV-2 genomes a week. Take any common mutation and follow its frequency in two regions: while travel between them is suppressed the two frequencies wander apart, and once travel resumes they converge again. Nobody recorded the journeys that caused the convergence, yet their imprint was left for us to read.

Whether that imprint is there at all depends on which mutations we choose to follow, and the ones tracked here are the ones that do not matter. A neutral allele is neither favored nor purged, so the only thing that moves its frequency in a region is the arrival and departure of the hosts carrying it. That is what makes it readable. Selected mutations have instead their own additional source of change: their frequencies move in a closed population as well, and once a rise can be produced from within there is no way to tell it from a rise produced by arrivals. What follows is the single linear equation that turns this convergence into an estimate of who moved where, and its application to two datasets with almost nothing in common beyond that neutrality: weekly SARS-CoV-2 genomes across regions, and ancient human genomes sampled centuries apart.

## A Metapopulation of Neutral Alleles

We consider a population divided into \(n\) subpopulations (regions, districts, age groups, etc) and follow the frequency \(X_i(t)\) of one neutral allele in each. Under neutrality the frequencies at a later time depend linearly on the frequencies at an earlier one:

\[ X_i(t+\Delta t) = \sum_{j=1}^n A_{ij}\, X_j(t) + \eta_i \]

where \(A\) is a right-stochastic matrix, its elements non-negative and summing to one within each row, and \(\eta_i\) is genetic drift, about which we need to know nothing at all except that its expectation vanishes. The deterministic term is linear because under neutrality the frequency of any union of lineages must obey the same evolution equation, and because the conditional expectation of a neutral frequency is extensive: starting with half as many mutants is expected to lead to half as many mutants later. The coefficient \(A_{ij}\) is the proportion of individuals in population \(i\) that originated from population \(j\) during the interval, the fraction of \(i\) replaced by migrants from \(j\). Read epidemiologically it is the proportion of infections that \(i\) imports from \(j\), which is why the COVID work{{cite: okada2025}} calls \(A\) the importation-rate matrix and the ancient DNA work{{cite: isacchini2026}} the migration matrix.

The same matrix can be read in the opposite direction of time, and that reading is worth having. Take a genome sampled in population \(i\) and follow its lineage of ancestors into the past. Each row of \(A\) sums to one over non-negative entries, which makes it a probability distribution over source populations, so \(A_{ij}\) is the probability that the lineage sat in \(j\) one interval earlier and \(A_{ii}\) the probability that it stayed put. Forward in time \(A\) is a table of migrant proportions; backward in time it is the transition matrix of a Markov chain on demes, and a genealogy is one walk of that chain.

If we subtract \(X_i(t)\) from both sides, the mechanism is in plain view:

\[ X_i(t+\Delta t) - X_i(t) = \sum_{j=1}^n A_{ij}\,\bigl[X_j(t) - X_i(t)\bigr] + \eta_i \]

where the bracket shows that \(j\) influences \(i\) only when the two frequencies differ, and that a larger \(A_{ij}\) makes \(X_i\) converge on \(X_j\) faster. Migration is a spring and drift is the noise that keeps it from ever settling.

## Two Impossible Datasets

The model is the easy part. What separates one application from the next is which statistic of the data survives the noise, and the two datasets we are comparing fail in opposite directions.

SARS-CoV-2 surveillance is dense in time and thin in loci. England was sequenced weekly across nine regions, so \(\Delta t\) is one week and there are dozens of consecutive time points, but a 30-kilobase genome under a strong clonal background offers only a few tens of approximately independent segregating alleles. Ancient DNA is the mirror image. The Allen Ancient DNA Resource{{cite: mallick2024}} genotypes individuals at up to 1.23 million positions, which is an enormous number of loci, and delivers them as a handful of skeletons per 300-year window, pseudohaploid, with the great majority of positions missing in any one sample and the dates themselves uncertain by centuries.

## Frequencies We Can Trust

With frequencies in hand the estimator writes itself. Minimizing the squared difference between predicted and observed frequencies over all right-stochastic matrices gives

\[ A^{\rm (LS)} = \operatorname*{arg\,min}_{A} \sum_{i,\,\mu,\,t} \Bigl[ X^\mu_i(t+\Delta t) - \sum_j A_{ij} X^\mu_j(t) \Bigr]^2 \]

where \(\mu\) labels the independent loci and the minimization runs subject to \(A_{ij}\ge 0\) and \(\sum_j A_{ij}=1\). Summing over a window of time points regularizes the fit and sets the resolution at which \(A\) is allowed to vary.

The estimator wants the true frequencies and we only ever see a noisy sample of them. The remedy in the COVID work is to stop treating what we measured as the state of the system. A hidden Markov model keeps the true frequencies as hidden states and gives genetic drift and sampling their own noise terms, with drift variance set by an effective population size and sampling variance set by the number of sequences and a per-region overdispersion. Gaussian noise makes the model a Kalman filter with an analytic likelihood, and the parameters follow from MCMC or, more cheaply, from expectation-maximization. Figure 2 shows the structure of this model.

{{figure: gene-flow-figKalman}}

## Distances We Can Trust

None of that survives contact with ancient DNA. A few genomes in a 300-year bin do not determine a frequency, the noise at that depth is not remotely Gaussian, and each sample covers a different subset of sites. We give up on individual frequencies instead, and write the dynamics directly in terms of quantities that many loci can estimate together.

The \(F_2\) statistic{{cite: patterson2012}} is the simplest of them: the squared difference in allele frequency between two populations, averaged over the whole genome. No single site is measurable at this sample size, but the average over a million of them is, and because sites enter that average independently, samples covering different subsets of positions still combine into one number. Under neutrality its expectation grows with the drift that separates two populations, which is why the field reads it as a genetic distance and builds admixture graphs out of it.

Write \(F_{ik}=\langle (X_i-X_k)^2\rangle\) for that distance and \(F'_{ik}=\langle (X'_i-X_k)^2\rangle\) for its lagged version, where the prime denotes time \(t+\Delta t\) and the brackets are the average over sites. The lag is the addition here, and it is what tells a donor from a recipient.

Figure 1 runs both statistics in the browser for two demes. The top panel is the relaxation of the model itself, neutral frequencies in \(A\) and \(B\) pulled together by migration, and the bottom panel is the distance between them with its two lagged versions. Symmetric migration pulls all three curves down together. Turn on flow in one direction only and the two lagged statistics separate, because only the receiving population has moved toward where the donor used to be.

{{figure: gene-flow-figRelax}}

We start from an identity that costs nothing:

\[ X'_i - X_k = (X'_i - X_i) + (X_i - X_k) \]

where the first bracket is how far population \(i\) moved and the second is how far apart the two populations were. Squaring and averaging over loci, we obtain

\[ F'_{ik} = F'_{ii} + F_{ik} + 2\,\bigl\langle (X'_i-X_i)(X_i-X_k)\bigr\rangle \]

where \(F'_{ii}\) is the mean squared displacement of population \(i\) over the interval and the cross term is the only piece that knows anything about migration. Substituting the relaxation form of the model, and using \(\sum_j A_{ij}=1\) together with the identity \(2\langle (X_j-X_i)(X_k-X_i)\rangle = -F_{jk}+F_{ji}+F_{ki}\), we find that the cross term becomes \(-F_{ik} + \sum_j A_{ij}(F_{jk}-F_{ji})\) and the equation collapses to

\[ F'_{ik} - F'_{ii} = \sum_{j=1}^n A_{ij}\,\bigl(F_{jk} - F_{ji}\bigr) \]

where every quantity on both sides is a measurable genetic distance and the dependence on \(A\) is still linear. Genetic drift has vanished. Its variance lives in \(F'_{ii}\), which enters \(F'_{ik}\) identically and cancels in the difference of the two lagged statistics.

Figure 3 is the same statement drawn as a triangle. The displacement of population \(i\) over one interval decomposes into a pull toward the other populations and a random kick, and only the pull is correlated with the direction of \(k\). Subtracting \(F'_{ii}\) discards the squared length of the displacement, drift included, and keeps its projection.

{{figure: gene-flow-figTriangle}}

Fixing \(A\) now takes \(n\,(n-1)\) equations for its \(n\,(n-1)\) free entries, and by stacking several time slices we make the system overdetermined and the fit stable. Each row is estimated on its own by non-negative least squares under a simplex constraint, which is the same constrained regression as before applied to different inputs. We handle the remaining sampling noise by subtraction rather than by modeling, using the unbiased estimator

\[ \hat{F}_2(A,B) = (a-b)^2 - \frac{a(1-a)}{N_A-1} - \frac{b(1-b)}{N_B-1} \]

where \(a\) and \(b\) are the observed allele frequencies and \(N_A\), \(N_B\) the numbers of observed alleles, so that the binomial variance contributed by finite sampling is removed analytically at every site.

Figure 4 puts the two estimators side by side on the same simulated history. Three demes evolve under a known importation matrix whose off-diagonal entries run from 2% to 7% per step, and we then choose how deeply to sequence and how many loci to follow. At shallow sampling the direct regression lifts every small coupling toward \(1/n\), exactly the bias described above, and piling on more loci does not rescue it: averaging a biased quantity leaves the bias. The \(F_2\) route stays near the truth and improves with loci. Push the sequencing depth up and the two converge, which is the regime the COVID data actually sits in.

{{figure: gene-flow-figEst}}

## Constructing Neutrality

Every equation above assumed neutrality, and neutrality is not something a dataset hands over. It has to be built. The COVID analysis works inside a single variant of concern and tracks lineages within that clonal background, so that the selective sweep between variants cannot masquerade as mixing, with clustering used to pick a set of alleles that behave neutrally with respect to one another. The ancient DNA analysis prunes sites in linkage-prone regions, drops low-frequency variants, and leans on the recent finding that allele frequency change in Europe over the last five millennia is largely explicable by gene flow without invoking detectable selection.

The failure mode is worth stating plainly, because it is the one that would break everything. A selected allele rising in two regions at once produces convergence that has nothing to do with migration, and the method will report it as an importation rate. The reading is only as good as the neutrality behind it.

## Rates Instead of Proportions

The appeal of this construction is how little it assumes. There is no phylogeny to reconstruct and no ancestral source population to postulate, which matters because both are modeling choices that quietly determine the answer. What comes out is a matrix of rates over an interval, estimated by a convex program that scales to whatever sequencing produces next.

Rates also generalize the quantity the field already measures. Multiplying the matrices over successive intervals, \(A(t)\,A(t-1)\cdots A(1)\), gives a matrix whose rows are the expected proportions of ancestry in each population traceable to each population at the starting time. That is precisely an admixture proportion, with the restriction to a single static episode lifted. Admixture proportions answer where ancestry came from. A product of stochastic matrices answers when it moved, and by which route.

## The Route into Britain

The route is where this stops being bookkeeping. Applied to nine Western Eurasian regions between 4000 and 1000 BCE, the reconstruction recovers the arrival of Steppe-related ancestry in Britain in the third millennium, but not as a direct delivery: the inflow into Britain comes overwhelmingly by way of Central Europe, and the corresponding peak in Central Europe East precedes it by several centuries. Groups expanding westward picked up ancestry from the farmers already living along the route, and by the time they reached the North Atlantic they were substantially admixed. A static ancestry proportion would show the Steppe component and stop there.

It would be dishonest to end on that without saying how thin the record still is. Outside the window from 3500 to 2000 BCE the sampling in some of those regions is too sparse for the interpolation to mean much, an apparent westward influx around 3500 BCE looks more like an artifact of poorly differentiated Neolithic populations than a migration, and the error bars on the Pontic Steppe are wide enough to cover a great deal. The method reads a signal that is genuinely there in the mutations nobody selected for. How far back it can be read is a question about excavation, not about inference.

## References

1. {ref: okada2025} [Okada et al., *PNAS* **122**, e2500663122 (2025)](https://doi.org/10.1073/pnas.2500663122), the SARS-CoV-2 transmission networks, with a [commentary](https://doi.org/10.1073/pnas.2533093123) in the same journal. Code at [Hallatscheklab/NetworkInfer](https://github.com/Hallatscheklab/NetworkInfer).
1. {ref: isacchini2026} [Isacchini et al., *bioRxiv* (2026)](https://doi.org/10.64898/2026.03.12.710875), the ancient DNA analysis and the \(F_2\) formulation used here.
1. {ref: mallick2024} [Mallick et al., *Scientific Data* **11**, 182 (2024)](https://doi.org/10.1038/s41597-024-03031-7), the Allen Ancient DNA Resource, released through [Harvard Dataverse](https://doi.org/10.7910/DVN/FFIDCW).
1. {ref: patterson2012} [Patterson et al., *Genetics* **192**, 1065 (2012)](https://doi.org/10.1534/genetics.112.145037), the \(f\)-statistics.

Elsewhere on this site, on inference that assumes as little as possible: [maximum entropy and density ratio estimation](me.html).
