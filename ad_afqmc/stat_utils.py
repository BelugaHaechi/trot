from functools import partial

import numpy as np

print = partial(print, flush=True)


def blocking_analysis(weights, energies, neql=0, printQ=False, writeBlockedQ=False):
    """
    Estimates the ph-AFQMC energy and the uncertainty of the estimate by blocking analysis. Context:
    - In ph-AFQMC, we hope to estimate the converged energy and its error (SD) after sufficiently many steps.
        - We first discuss the existence of this converged energy and converged energy error.
        - For some initial time, the expected energy evolves away from the initial guess energy (step 0).
            - typically the energy decreases rapidly, bc the initial guess is typically higher than E0
        - In fp-AFQMC, this expected energy is theoretically known to converge to the true E0.
            - but, the variance is empirically known to diverge exponentially! (sign problem)
        - In ph-AFQMC, this expected energy seems to empirically converge to a biased Eph near E0.
            - the variance also empirically seems to converge to some small Vph (no divergence!)
            - the bias is a result of the phaseless constraint as a tradeoff for this niceness
            - **SUMMARY: after some cutoff time, ph-AFQMC behaves like a weakly stationary process**
                - constant mean, constant variance, covariance depends on time gap only
                - there is NO proof of this, purely empirical
    - So after the initial time (named "equilibration phase"), can we estimate Eph using hat(Eph), and also estimate Var(hat(Eph))?
        - Simplest idea: just stop once we leave the eql phase, and analyze the walkers at the final step.
        - Result: get sample mean and sample variance that estimate Eph and Vph.
            - each observation is iid in vanilla AFQMC (not anymore with SR and ph)
            - so sample mean is an unbiased estimator of Eph with variance Vph/N
            - N/(N-1) sample var is unbiased estimator of Vph (basic result from statistics!)
                - (NOTE: its sqrt is actually a slightly biased estimator of sqrt(Vph), this is subtle)
            - then we can estimate variance of sample mean (theory: Vph/N) as (sample var / (N-1))
        - To do better, we can simply increase the number of samples.
            - Issue: we have to propagate all of them past the eql phase. Costly!
                - Say the eql phase has length K. Then per sample cost is K steps.
                - Can we do better?
    - We can get effective samples more efficiently by taking advantage of the weakly stationary property of ph-AFQMC.
        - Idea: As observations are further apart, their autocorrelation (covariance) decreases empirically
            - math: Cov(X_t, X_u) decreases as |t-u| increases
            - If we allow a walker to propagate well beyond the eql phase, and only observe it every L steps,
            then as L increases, the observations will seem more and more independent!
            - Also, if we consider the trajectory of an entire walker, and average consecutive observations of length L,
            then as L increases (the blocks get larger), and then the "block averages" themselves will be more and more independent!
                - math: define B1 = average(X1 + ... + XL), B2 = average(X_L+1 + ... + X_2L), so on.
                - then Cov(B_t, B_u) decreases as L increases!
                    - idea: only obs near the boundary have appreciate Cov, and they are a minority
                    - despite the fact that the steps within the blocks are still correlated!
        - This "blocking" idea is useful because as the block averages approach iid as L increases, we can now view
        each block average using the iid results from above!
            - That is, each "block average" is effectively an iid sample (like a new walker) for long enough L
            - now the average of a sample of N block averages is equal to the average of all the NL observations
                - bc we assumed weak stationarity, this big average is still an unbiased estimator of Eph :)
                - however, the variance of the sample mean of NL obs is not Vph/(NL-1) because of the autocorrelation
            - now define the variance of the sample of N blocks averages as Vb
                - the nice thing: the variance of the sample mean of blocks, is indeed Vb/N, as blocks are ~iid!
                - so the correct estimate for the variance of the average of a sample of block averages is Vb/(N-1)
                - and as the avg of a sample of block avgs is equiv to the avg of all observations (before blocking)
                - we get the var of the avg of all observations to be estimated by Vb/N. Done!
                - so we get Var(hat(Eph)) estimated by Vb/(N-1), instead of the wrong Vall/(NL-1)
                    - this is unbiased in the limit that L is large enough
                    - ok actually there is another detail involving V1 - V2/V1 in the denominator for weighted variance
            - OK I admit this is A LOT of symbols. We need to be VERY careful to distinguish all the subtlety!
                - when I get a chance I should write this down. it was a lot more complicated than I thought
                - but the result is that, **blocking analysis is the correct way to go**
                    - with sufficiently large L (chosen by the "plateau" criteria: when increasing L only changes result by <5%)
                - bc of autocorrelation, no blocking (L=1) will underestimate the actual variance
                - by increasing L, the computed variance estimate will approach the true variance asymptotically
                - then we can set a L cutoff so that the variance estimate is good enough
                    - NOTE: the tradeoff of a large L is that, Var(B) is no longer accurately estimated by sample var of B
                        - this accuracy is described by Var(sampleVar(B)) which depends on 4th moment of B
    - All this to get a good estimate of the variance of the estimate of the mean of a weakly stationary process!
        - and this weakly stationary process is an approximation of the long-time behavior of an approximation of fp-AFQMC
        - and fp-AFQMC itself is a stochastic process... OK
    """
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    meanEnergy = weightedEnergies.sum() / weights.sum()
    if printQ:
        print(f"#\n# Mean: {meanEnergy:.8e}")
        print("# Block size    # of blocks         Mean                Error")
    blockSizes = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000])
    prevError = 0.0
    plateauError = None
    for i in blockSizes[blockSizes < nSamples / 2.0]:
        nBlocks = nSamples // i
        blockedWeights = np.zeros(nBlocks)
        blockedEnergies = np.zeros(nBlocks)
        for j in range(nBlocks):
            blockedWeights[j] = weights[j * i : (j + 1) * i].sum()
            blockedEnergies[j] = (
                weightedEnergies[j * i : (j + 1) * i].sum() / blockedWeights[j]
            )
        v1 = blockedWeights.sum()
        v2 = (blockedWeights**2).sum()
        mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
        error = (
            np.multiply(blockedWeights, (blockedEnergies - mean) ** 2).sum()
            / (v1 - v2 / v1)
            / (nBlocks - 1)
        ) ** 0.5
        if writeBlockedQ:
            np.savetxt(
                f"samples_blocked_{i}.dat",
                np.stack((blockedWeights, blockedEnergies)).T,
            )
        if printQ:
            print(f"  {i:5d}           {nBlocks:6d}       {mean:.8e}       {error:.6e}")
        if error < 1.05 * prevError and plateauError is None:
            plateauError = max(error, prevError)
        prevError = error

    if printQ:
        if plateauError is not None:
            print(f"# Stochastic error estimate: {plateauError:.6e}\n#")

    return meanEnergy, plateauError


def reject_outliers(data, obs, m=10.0):
    d = np.abs(data[:, obs] - np.median(data[:, obs]))
    mdev = np.median(d) + 1.0e-10
    s = d / mdev if mdev else 0.0
    return data[s < m], s < m


def jackknife_ratios(num: np.ndarray, denom: np.ndarray):
    r"""Jackknife estimation of standard deviation of the ratio of means.

    Parameters
    ----------
    num : :class:`np.ndarray
        Numerator samples.
    denom : :class:`np.ndarray`
        Denominator samples.

    Returns
    -------
    mean : :class:`np.ndarray`
        Ratio of means.
    sigma : :class:`np.ndarray`
        Standard deviation of the ratio of means.
    """
    n_samples = num.size
    num_mean = np.mean(num)
    denom_mean = np.mean(denom)
    mean = num_mean / denom_mean
    jackknife_estimates = np.zeros(n_samples, dtype=num.dtype)
    for i in range(n_samples):
        mean_num_i = (num_mean * n_samples - num[i]) / (n_samples - 1)
        mean_denom_i = (denom_mean * n_samples - denom[i]) / (n_samples - 1)
        jackknife_estimates[i] = (mean_num_i / mean_denom_i).real
    mean = np.mean(jackknife_estimates)
    sigma = np.sqrt((n_samples - 1) * np.var(jackknife_estimates))
    return mean, sigma
