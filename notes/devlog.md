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

after done:
read output from memwatch.py, note Peak PSS
to generate graph:
python /home/dsl44/trot/plot/memory.py <PID>

Tip: can replace the independent (experimental) variable with 
`int(sys.argv[1])` to conveniently specify it through command line.


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


## 11/19, 11/20
4. Extend walker number scaling argument
- let's go back to H2, try different ckpt intervals
- aim is to have more lines in the graph instead of just 400 and 50

Done. Tested 50,100,200,400,600,800 walkers for H2 STO-6G
For checkpointing freqs 1,2,5,10,25,50,100,200,400,None(8000)
There were also 2 types of 50, because the E-block has length 50, you can either checkpoint before the energy sampling postprocessing step (within the E-block), or checkpoint the entire E-block (after energy sampling).
Obviously the former would take more memory because of the relatively large intermediates during postprocessing. We call this case 50dt (or 50b).
Still, this comparison is useful because then we can see exactly how much memory is used by saving postprocessing residuals, which is always done for 1,2,5,10,25 cases (because shorter than 50). This partially explains why these are quite memory-heavy and less optimal compared to 50.

In this process, pre-E checkpointing used to be even more expensive!
For 800 walkers, used 4+ GB more than post-E. To find out why, I looked at saved residuals, and found that the culprit was that it was saving all of the random fields, which is even more expensive than the walkers. This cut 1GB from each process.
Perhaps this can be further optimized, because dt50 additionaly saves
c128[20,8,1,800,18,1,1]
I wonder where this 18 (Cholesky) is coming from...
OK some discussion with Chat suggests that it might be from `trial.calc_overlap_vmap`. Resonable, should try checkpointing that as well.

Made a nice graph called memory-ckpt-nw. Hid 10 and 50b because they were very close together along with 25. Notably no checkpointing uses so much memory that it has to be put in a subplot, which is good because it demonstrates how checkpointing is essential.





FUTURE: Extract wall time information as well. Use statistical averages for EQ and SP steps, we have 10 and 4 samples. The SP is important.
This would be annoying to do by hand, so make a throwaway code for it.
Add try catch because sometimes the out files could be corrupted.

FUTURE: Benchmark on system size.

FUTURE: Analysis and reverse engineer scaling of A and B.
Derive optimal adaptive strategy. Verify.

FUTURE: Use code to attempt larger systems as showcase. For publication.


## 11/22

Was looking at saved residuals in CS50, thinking about the remaining discrepancy between CS50 and CE1. Even after cutting 1GB off (no more saving random fields), there is still a 0.5GB gap (3.0GB vs 2.6GB).

Turns out it could be optimized further, see dt50' and dt50".
Basically the energy evaluation is the second most expensive task, and it could also be checkpointed. (dt50')
Furthermore, the orthonormalization and overlap calculations are the third most expensive (adds constant to leading term), so if we checkpoint all 3 postprocessing steps, we save the most memory. (dt50")

Technically, to make it a fair comparison between CS and CE, I should rerun the computations with dt"...
FUTURE: Run CS" 1/2/5/10/25/50, which is a total of 36 runs... at this point we need to automate this

But also, it's not a priority. Let's do other experiements first before coming back to fix this.


## 11/23

Gathering walltime data from the previous experiment ckpt-vs-nw. Made graph for it as well.

Found anomaly with 8000/600, it was much slower than even 8000/800, turned out it because of OOM and it was using swap space
which is really slow. TODO: rerun 8000/600 with larger memory node, when available

also rerun 50/800, shows a diff trend than expected

Then, move on to system size. Time to make the automation

## 11/24 Automation

Created bench/ series of scripts, moving hchain and memwatch here.
Made hchain fully command line, and integration with CSV (launch) and SLURM (batch)

USAGE

To run from interactive:

python bench/launch.py bench/hchain.py jobs/JOBNAME.csv ID

To submit async to scheduler:

sbatch bench/batch.sh bench/hchain.py jobs/JOBNAME.csv
and a bunch of SLURM settings

The script automatically hooks up the conda env and python path. However, this would also require manually setting SLURM settings.

Improved architecture: an extra script, submit.py
Automatically derives most of the SLURM settings from the CSV and partition name, assembling the full SLURM cmd

python bench/submit.py bench/hchain.py jobs/JOBNAME.csv PARTITION 4 --dry-run

So the overall hierarchy:

hchain.py, memwatch.py - run the benchmark test once
launch.py - auto supply params from CSV
batch.sh - auto setup and schedule tasks in CSV
submit.py - auto create the batch command


Also supports list of partitions now. (Simple impl)

OK my AI was able to give a very good summary, after reviewing it here it is:

### 1. Benchmark automation

Developed a modular benchmarking framework:

- `hchain.py`: standalone AFQMC calculation with named CLI arguments and centralized defaults.
- `memwatch.py`: automatic process-tree memory profiling.
- `launch.py`: CSV-to-CLI translation, working directory creation, and concurrent execution of calculation and memory monitor.
- `batch.sh`: Slurm execution environment setup.
- `submit.py`: automatic generation of `sbatch` commands, job names, array ranges, and resource allocations.

Repository organization:

- `bench/`: benchmark and automation scripts.
- `jobs/`: experiment specifications (CSV).
- `jobs/.slurm_logs/`: scheduler and execution logs.
- `runs/`: generated calculation results.
- `plot/`: analysis and visualization.

Each run automatically produces:

- `options.json`: fully resolved calculation parameters.
- `memlog.csv`: time-resolved RSS and PSS measurements.
- `memtot.json`: peak total RSS/PSS, elapsed time, and per-process peak PSS.
- `.out`: combined scientific stdout and stderr.

Additional features:

- Multi-partition submission.
- Resource defaults from a local partition CSV, using the first specified partition.
- Dry-run support (`-n`).
- Automatic creation of output directories.
- Boolean CLI argument support.

The framework is largely independent of AFQMC and can be reused for other computational projects.

### 2. MPI scaling experiment

Investigated the trade-off between MPI parallelism and replicated memory overhead.

Settings: H2/STO-6G, CE50, 400 total walkers.

| MPI ranks | Walkers/rank | PSS (GiB) | RSS (GiB) | avg/eq (s) | avg/sp (s) | Total (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 0.787 | 0.890 | 1.961 | 47.288 | 255.031 |
| 2 | 200 | 1.219 | 1.492 | 1.429 | 28.660 | 139.535 |
| 4 | 100 | 2.128 | 2.747 | 1.049 | 16.725 | 88.459 |
| 8 | 50 | 3.849 | 5.165 | 0.812 | 7.810 | 50.048 |

Observations:

- MPI provides substantial coarse-grained parallelism, contrary to the initial expectation that a single JAX process would be sufficient.
- Eight ranks achieve a 6.05x AD speedup at a 4.89x memory cost.
- Replicated memory overhead is approximately 0.44 GiB per additional rank.
- Larger walker batches exhibit better per-walker computational efficiency: 400 walkers in one rank take 47.29 s, compared with 8 x 7.81 = 62.48 s.
- MPI parallelism and JAX batching are complementary.
- Discovered that the original implementation explicitly restricts CPU multithreading via XLA flags. The experiment therefore does not represent unrestricted JAX multithreading.

**Decision:** Retain four MPI ranks for the walker-population benchmarks. Reconsider the optimal rank count for larger systems.

#### Memory monitoring bug under Slurm

Initially, memwatch reported approximately 0.21 GiB regardless of MPI rank count.

Hydra was launching MPI processes outside the monitored process tree under Slurm.

Fixed by setting:

`export HYDRA_LAUNCHER=fork`

This restores local process ancestry for single-node execution, allowing memwatch to capture the MPI workers.

### 3. Generalized checkpointing implementation

Removed hardcoded checkpointing intervals and introduced configurable checkpointing through the propagator dataclass.

Parameters:

- `n_steps_per_ckpt`
- `n_ene_per_ckpt`
- `do_sr_ckpt`
- `do_eval_ckpt`

Implementation changes:

- Pass checkpointing options from `mpi_jax.py` into the propagator.
- Consolidate repeated propagator constructor arguments.
- Validate nonnegative checkpoint intervals and divisibility in `__post_init__()`.
- Preserve the original random field generation when CS is disabled.

CS requires a different random field generation scheme, generating fields within each checkpoint rather than across the entire E-block. Consequently, different CS intervals may produce different random trajectories.

#### Improved CS″ strategy

Combined walker orthonormalization, overlap evaluation, and local energy evaluation into one checkpointed function.

Evaluation checkpointing is automatically enabled whenever CS is enabled. It can also be independently enabled for CE or CSR.

Condition:

`if propagator.n_steps_per_ckpt or propagator.do_eval_ckpt:`

Previous measurements at 800 walkers/rank:

| Strategy | Peak PSS (GiB) |
|---|---:|
| Original CS50 | 3.00 |
| CS50 + local energy checkpoint | 2.65 |
| CS50″ | 2.57 |
| CE50 | 2.51 |

Plain CS is inefficient because evaluation residuals accumulate across all E-blocks. Combined evaluation checkpointing replaces these large residuals with relatively compact input states.

### 4. Explicit JAX compilation and runtime methodology

Identified systematic timing bias in the first sampling iteration and first two equilibration iterations.

Implemented explicit compilation using `.lower().compile()` for both equilibration and differentiated sampling.

Moved compilation and JAXPR/saved-residual diagnostics before the iteration-0 timestamp.

This separates compilation costs from recurring execution costs without discarding timing observations.

#### Propagation state bugs

Explicit compilation exposed two inconsistencies previously hidden by automatic JIT recompilation:

1. `n_killed_walkers` was absent from the initial propagation state but introduced after the first iteration, changing the input pytree structure.
2. The same variable changed from an integer count to a floating-point fraction after normalization.

Fixed by initializing the variable with a consistent floating-point dtype.

The propagation state must preserve its pytree structure, shapes, and dtypes between calls to the compiled executable.

### 5. Second-generation walker-population benchmark

Prepared jobs/ckpt-vs-nw-v2.csv and submitted the rerun series. Improvements:

Improved CS″ checkpointing: Combined orthonormalization, overlap calculation, and local energy evaluation into a single checkpoint, preventing large evaluation residuals from accumulating across E-blocks.

Generalized checkpointing implementation: All strategies are selected through configuration parameters rather than manual source-code changes, ensuring a consistent implementation across experiments.

Consistent propagation state: Fixed changes in pytree structure and dtype that previously caused unnecessary JIT recompilation during equilibration.

Improved timing statistics: Compilation and diagnostic costs are excluded from iteration timings, allowing all timing observations to be retained and yielding substantially narrower confidence intervals.

Automated memory profiling: Consistent process-tree PSS and RSS measurements, including all MPI workers, with time-resolved logs and per-process peak memory.

Additional checkpointing strategy: CSR400 is now included, allowing direct comparison with CE400.

Improved reproducibility: Explicit experiment parameters in CSV files, resolved parameter dumps in JSON, isolated working directories, and standardized execution environments.

Correction of anomalous measurements: Rerun previously affected configurations, particularly the uncheckpointed high-population runs that experienced swapping and the anomalously slow CE50/800 run.

### 6. TODO

Verify completion and correctness of the new benchmark series.

Aggregate memory and runtime results.

Regenerate memory and runtime scaling plots.

Compare CS″, CE, CSR, and uncheckpointed AD.

Identify optimal checkpoint intervals.

Begin system-size scaling using hydrogen chains.

Investigate nested checkpointing for larger systems if necessary.

Clarify computing-resource authorization after break.

Eventually implement structured runtime output and a generic results collector.


## 11/25

Obtained rerun results. Plotted new graphs. There are some new findings!

Because we have a lot of data, I created two versions of graphs: full and clean. The clean version is less cluttered.

In terms of memory, the newly improved CS" strategy outperforms CE50! Only slightly, but the growth rate is a lot slower.

Memory growth rate: MiB per walker/proc
None    47
CS400   2.73 (17x reduction)
CE50    0.45 (6x reduction on top)
CS10"   0.24 (1.9x reduction on top) total reduction: ~200x

Note: CS without eval ckpt wouldn't be this efficient!

Walltime: trend no longer that clear
- Generally, reverse takes 2~3x the time of forward
    - CS is 2.5x consistently
    - None is 2.4x, CE50 is 2.5x, CE200 is 2.9x, CSR400 is 3x

- CS checkpointing always has slower forward than no checkpointing (!)
    - EXCEPT CS50, which is similar to CE/None
    - CE checkpointing has the same forward time (basically indistinguishable)
    - CS is slower probably because of the fragmented random field generation
        - Except CS50, which generates together again!

- CS checkpointing has a slower reverse than no checkpointing
    - Looking at the full graph, the progression is like
        - CS: 1 < 2 > 5 > 10 > 25 > 50
        - CE: 50 < 100 = 200 = 400 < CSR400
        - None less than all

    - For CS, increasing the interval improves runtime, from +58% down to +7%
        - Except CS1, which is about as fast as CS10 (+44%)
    - For CE, the interval seems to have no effect, and is as slow as CS25 (+20%)
        - Except CE50, which nearly as fast as None (+8%)
    - CSR400 seems to introduce further overhead (+30%)

    - Hypothesis for CS1 and CE50
        - Checkpointing at natural algorithmic boundaries allows better compilation


Plotted several more graphs. Notably:
- Relative walltime increase. This graph is probably easier to read, shows that the multiple is about constant, between 2 and 4.
- Relative memory reduction factor.
    - Original graph wasn't as good, as it shows that the reduction factor increases!
        - This is because of the large constant overhead that get diluted
    - Intead, plotted growth factor reduction from linear regression. No distraction by the constant
        - Wow, 150x reduction in total! (Not 200x because of regression)
            - Rerunning the w200 still gave outlier (unexpectedly low, but that should inflate our reduction factor)
- Tradeoff comparison: relative wall time increase vs relative memory reduction
    - Very beautiful, illustrates the retrograde behavior of the optimization


At this point, we are ready to proceed to system size scaling.


## 11/25


