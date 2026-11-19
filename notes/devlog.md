## Phase II work on memory efficiency

Recap of Phase I
- we did a thorough study of the current AFQMC theory
    - realized that ph-AFQMC has the core assumption that there exists a converged distribution (weakly stationary distribution), which seems to be true empirically
    - this can be used to sample arbitrarily long lengths
- we did a full read of the code in ad_afqmc, and added commentary to most of the functions
- we made a reusable, single-run test called `grad/hchain.py`
    - can do SR-ph-AFQMC@DF-HF/sto-6g with AD gradients for any H-chains of equal spacing
        - supported gradients: any 1-rdm operator, and any 2-rdm operator
        - therefore indirectly supports nuclear gradients via rdms
        - but nuclear gradients of rdms are FD only, done pre-AFQMC
    - can include trial response, via built-in HF SCF solver
    - can have gradient checkpointing
    - can set a custom equilibration phase
- we ran 66 tests on my own computer, each 4*100 walkers (4 processes)
    - found: increasing total number of samples decreases the error of E and F
    - found: having AD length too small would cause F to become biased
    - found: having equilibration length too small would include initial biased E values
    - found: safe AD length is approximately the same as equilibration length
        - unknown: whether this varies wrt system size
    - found: previous checkpointing frequency was not optimal, near sqrt frequency is better
        - not very noticeable at first, but scales 4 times slower wrt walker count (linear), maybe 3x wrt system size (nonlinear)
        - unknown: more testing needed for clearer scaling rules
            - idea: do 1 process only, and use large-mem mei resources
            - idea: do walker count first, then system size, then walker count on larger system. (maybe like 5 parallel lines wrt walker count, each line a larger system)
        - unknown: scaling wrt other AD intervals !!!!!


OK we got some work to do, and it's not too complicated, we can do this!

- with this data, we can start thinking about a and b
    - we'd like to see that as system size changes, a and b can make the naive sqrt not optimal


### Parameters to reproduce Jo's paper Sec. III.E

System
- basis:            sto-6g
- bond length:      1.6 au (close to eql)
- density fitting:  CDERI form

AFQMC
- 1p basis:         Lowdin orthon AO
- tot walkers:      200                 response paper: 12500 samples, 36 cores
    - processes:        20
        - OK my pc probably can't launch that many procs
    - walkers per proc: 10
- timestep:         0.01
- sr interval:      1.0
    - tradeoff: infreq sr = less biased but more uncertain

- total time:       25.0 response paper

FD Integral Derivs
- scheme:           central diff
- perturbation:     1e-5

Hardware:
- gpu:              nvidia A100
- cpu:              20-core Intel Xeon

Results:
- time cost (cpu):  8-10x   3-4x (response)
- time cost (gpu):  2-4x
- max size til oom: 84 H
- see fig 5 for scaling


## 10/9/2026 Scaling tests

Got new node on mei2_medmem, has 64g memory (62 actual usable)
Not the biggest node, but mei4 and stuff are occupied

We will be conducting tests with
- 1 process         ! to save memory
- 100 walkers
- 0.01 dt
- 0.50 E
- 4.0 SR
- 80.0 eq
- 80.0 AD x4

on system
- 1.8 internuclear sep (note: in H2, equ is 1.40 Bohr)
- 2 hydrogens
- sto-6g basis (2)
- default auxbasis (18)

should we enable rot?
- previously we didn't because the test didn't use it and we wanted to compare with it
- but that caused the problem where our numbers were less accurate when compared with high level methods
- if we enable rot here, then our results will not be comparable to any previous results we obtained over the summer

- so this is a distinction between **reproduction** and **main benchmark**
- in Phase I we were worried that our calculations won't agree with the previous person's setup, but we did verify that things were working
- now in Phase II we want to get the best performance of the method, so we shall use the more accurate settings!
    - while we can first run some tests that look like Phase I, so we can cross-compare

### Procedure to begin calculation

conda activate trot
export PYTHONPATH='/home/dsl44/trot'
cd trot/runs/MULTI/TIERED/RUN/NAME
python ~/trot/examples/grad/hchain.py > .out

Check .out file to confirm parameters.

in separate terminal on same machine:
top --> find PID of the python process
cd trot/runs/MULTI/TIERED/RUN/NAME
python /home/dsl44/trot/ad_afqmc/memwatch.py <PID>

back to the first terminal: input Enter to begin


## 10/9/2025 Verification of effects of "orbital rotation"

A better name would be "trial wfn relaxation", in short "TR"

Previously, we did not enable this option in order to compare with benchmark data.
Now, however, as we move on, we'd like to use the best possible parameters.

Unexpected: enabling orbital rotation has no significant effect for H2

Disabled, l80, t3200:   -0.11084(18)
Disabled, l80, t8000:   -0.11082(12)
Enabled, l80, t8000:    -0.11087(13)


What if we compare H10, instead of H2?

Disabled, l80, t8000:   -0.07879(13)
Enabled, l80, t8000:    -0.07899(13)

OK this difference is more noticeable, but is still within 2 sigma.

The wall time was similar for both methods. So moving forward, we should be comfortable with enabling TR.

Looking back at Jo's paper, it notes that TR has the biggest effect when bond length is farther from equilibrium. (Their example was CH in CH4, eql 1.086 Å, and the values looked fine)

Previously we have been using H2 with bond length 1.8 Bohr, while eql is 1.40 Bohr
OK, this was already quite non-equilibrium but let's try H2 with an extreme 5 Bohr

Same seed used.
Disabled, l80, t3200:   -0.03983(12)
Enabled, l80, t3200:    -0.03981(15)
No significant difference.

Energies also in agreement within 2 sigma.

However, the disabled case had this warning:
```python
/home/dsl44/trot/ad_afqmc/driver.py:1248: RuntimeWarning: invalid value encountered in divide
  avg_rdm1 = np.einsum(
/home/dsl44/trot/ad_afqmc/stat_utils.py:77: RuntimeWarning: invalid value encountered in scalar divide
  meanEnergy = weightedEnergies.sum() / weights.sum()
```

Interesting: Energies for HF and CCSD differ significantly.
Theoretically, CCSD is EXACT, at -0.94393
RHF yields -0.69478
ph-AFQMC yields -0.818~9, so it's between RHF and CCSD

Magnitude of gradients: CCSD is -0.0033 (!) which is weird given how nonequilibrium this is
OK maybe makes sense as the H are so far apart, that they don't repel strongly anymore
However, AFQNC and HF gradients (-0.055) are an order of magnitude larger!

Near equ our AFQMC gradients were close to CCSD, but now this gap is large!
Agrees with Jo's paper that the error gets large far from equilibrium. (but that was wrt FD hmm)

Because we used the same seed, and the energies came out different, we can be sure that TR was active. Looking at the implementation I don't see anything wrong.

**Thus I question the actual utility of TR.**
I am unable to reproduce the effect of TR being significant for AD gradients in H2 far-from-eql. In Jo's paper, as the FD-AD value is based on the same FD values while the AD has variants with and without TR, the difference in FD-AD values for these variants is the difference in AD variants.

Let's try more data points. 3 bohr

This is crazy, got exactly the same results??? How did this happen?
-0.11624(31) AFQMC
-0.1501 HF
-0.0705 CCSD

Again AFQMC gradient is between HF and CCSD.
Energy-wise, AFQMC agrees with CCSD much more.

Even more data points. 4 bohr, 1 bohr. This completes 1-1.8-3-4-5

Again, identical results for with and without TR, for both 1 and 4. ???

Something is probably wrong? Let's try to debug this and see if the right sampling procedure is enabled for each case.

Why???

possible guess: SCF not converged?

### 11/14 Verifying the correctness of TR code

Question:
1. Does the TR code do SCF correctly? (yield the right energy)
2. Are the results from TR and noTR really identical up to seed? 

r1.8, h2
PySCF RHF -1.08826645285862
AD-TR RHF -1.4405323794834757

TR:         -1.4442300167064859
wave_data:  -1.4442300167064865

seems to be OK?
so TR should work. the code:
```python
    from pyscf import gto, scf, lib
    import numpy as np

    rH = 1.8
    nH = 2
    atom_symbols = np.array(["H"] * nH)
    coords = np.zeros((nH, 3))
    coords[:, 2] = np.arange(nH) * rH

    atomstring = list(zip(atom_symbols, coords))
    mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
    mf = scf.RHF(mol).density_fit()
    mf.with_df.build()

    hcore = jnp.asarray(mf.get_hcore())
    enuc = jnp.asarray(mol.energy_nuc())

    # DF factors: (naux, nao, nao)
    cderi = np.asarray(mf.with_df._cderi)
    nao = mol.nao_nr()

    if cderi.ndim == 2 and cderi.shape[1] == nao * (nao + 1) // 2:
        cderi = lib.unpack_tril(cderi)
    elif cderi.ndim == 2 and cderi.shape[1] == nao * nao:
        cderi = cderi.reshape(-1, nao, nao)
    elif cderi.ndim != 3:
        raise ValueError(f"Unexpected DF factor shape: {cderi.shape}")

    L = jnp.asarray(cderi)

    nocc = mol.nelectron // 2
    def energy_tot(mo_coeff):
        C = mo_coeff[:, :nocc]

        # Spin-summed density matrix
        D = 2.0 * C @ C.conj().T

        # Coulomb: J[p,q] = sum_P L[P,p,q] * sum_rs L[P,r,s] D[r,s]
        rho = jnp.einsum("Prs,sr->P", L, D)
        J = jnp.einsum("P,Ppq->pq", rho, L)

        # Exchange: K[p,q] = sum_Prs L[P,p,r] D[r,s] L[P,s,q]
        K = jnp.einsum("Ppr,rs,Psq->pq", L, D, L)

        # RHF total energy
        E_one = jnp.einsum("pq,qp->", hcore, D)
        E_coul = 0.5 * jnp.einsum("pq,qp->", J, D)
        E_exch = -0.25 * jnp.einsum("pq,qp->", K, D)

        return jnp.real(E_one + E_coul + E_exch + enuc)

    debug.print('{}', energy_tot(mo_coeff))
    debug.print('{}', energy_tot(wave_data))
    # debug.print("{}\nmax {}", mo_coeff - wave_data, jnp.max(jnp.abs(mo_coeff - wave_data)))
```


Now Q2...
```
# dt: 0.01
# n_prop_steps: 50
# n_ene_blocks: 8
# n_sr_blocks: 4
# n_blocks: 40
# n_ene_per_sr_eql: 8
# n_sr_per_b_eql: 2
# n_eql: 1
# n_walkers: 100
# walker_type: rhf
# trial: rhf
# ad_mode: reverse
# do_grad: True
# orbital_rotation: True
# do_sr: True
# seed: 1239877
# symmetry: False
# save_walkers: False
# ene0: 0.0
# free_projection: False
#
Removed en_der_afqmc_0
Precomputing contractions and propagators...
Done. Initializing walkers...
# Equilibration sweeps:
#   Iter        Block energy      Walltime
#     0      -1.088266453e+00     1.96 
#     1      -1.114304781e+00     4.77 
#
# Sampling sweeps:
#  Iter      Mean energy        Stochastic error     Mean observable     Walltime
Using propagator: propagate_phaseless_nucgrad
     0                                                                       5.00
    4     -1.119662070e+00                    -      0.000000000e+00        23.6
    8     -1.119591385e+00      1.114298291e-03      0.000000000e+00        36.8
   12     -1.119318244e+00      8.293336448e-04      0.000000000e+00        50.0
   16     -1.119779943e+00      7.885842311e-04      0.000000000e+00        63.2
```

TR False
```
# Equilibration sweeps:
#   Iter        Block energy      Walltime
#     0      -1.088266453e+00     1.94 
#     1      -1.114304781e+00     4.63 
#
# Sampling sweeps:
#  Iter      Mean energy        Stochastic error     Mean observable     Walltime
Using propagator: propagate_phaseless_nucgrad_norot
     0                                                                       4.86
    4     -1.119662070e+00                    -      0.000000000e+00        22.5
    8     -1.119591385e+00      1.114298291e-03      0.000000000e+00        35.5
   12     -1.119318244e+00      8.293336448e-04      0.000000000e+00        48.4
   16     -1.119779943e+00      7.885842311e-04      0.000000000e+00        61.2
   20     -1.119690868e+00      6.583771557e-04      0.000000000e+00        74.1
```

OK now going back to the first time we used the same seed: rh5
TRon: -8.097098470e-01 at eql 10
noTR: -8.076201081e-01 at eql 10

now retry with TRon: -8.076201081e-01 at eql 10 (!)
so somehow the first time we tried TRon, the random numbers were used slightly differently, causing this to happen

*but otherwise the results from TR and noTR are identical*

this is understandable, because TR yields the same trial wavefunction, so the energy should at least be the same. but somehow the gradients are the same as well!

OK this is really confusing. I also confirm that I did not change the logic of the optimize_orbs function, nor the propagate_phaseless_nucgrad sampler.

We have also confirmed that our code has been selecting the correct sampler


OK at this point I think we should stop. Let's continue with using TR because
1. it is the theoretically more accurate thing to do
2. it does not subtantially add computation time
3. it does not change the computation results (so far)

and we really need to move on benchmarking for *scale*


## 11/14 Research questions

What we DON'T have time for: Accuracy claims other than those directly related to reverse differentiation (TR and AD length)
- Bond length, atom type
- Basis set, auxiliary basis set
- Step size, sampling frequency, SR frequency
- Equilibration duration (as long as sufficient)
- Sampling duration (just demo 4 AD blocks, little effect on peak memory)

What we DO care about: Peak memory cost vs scale
- System size: electrons
- Walker population per process
- Checkpointing frequency




1. Does TR affect accuracy? 
    Preliminary results: No, in fact somehow has no effect at all for cases tested: bond length 1,1.8,3,4,5 for H2, and H10 with r1.8

2. Does system size affect required equilibration time?
    Simple test: just try 100 hydrogens, and print out the equilibration process.
    Perhaps unrelated to number of basis functions. Likely only related to the 0-1 energy gap, because of the decay being exp(-t(E1-E0))
    For longer hydrogen chain, the gap decreases roughly as 1/N
    Assume that ph-AFQMC converges to the stationary distribution when
    exp(-t∆E) < a
    which is
    -t∆E < ln a
    which is 
    t > ln(1/a) / ∆E
    if for Hchain, ∆E ~ C/N
    then
    t > N ln(1/a) / C
    therefore equilibration time t proportional to N
    let's test this

    Preliminary results: NO! Thankfully
    - H100 energy settled in just t12, similar to H4 and H2
    - Conservatively speaking, perhaps it settled in t40
    - Still, very nice, don't need to worry about this

    *NOTE* For more rigorous results, need to look at shorter intervals... print out the first few steps. Perhaps H100 will spend slightly more time.

3. Does basis set size affect required equilibration time?
    Probably not at this point.
    H4 cc-pvtz: still settled in just t12, safely converged in t40
    OK great! So let's just move on to scaling.


## 11/19
4. Extend walker number scaling argument
- let's go back to H2, try different ckpt intervals
- aim is to have more lines in the graph instead of just 400 and 50
