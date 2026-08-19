from functools import partial
from functools import reduce
print = partial(print, flush=True)

import numpy as np
import jax.numpy as jnp

import pyscf
from pyscf import gto, scf, cc
from ad_afqmc import run_afqmc, grad_utils


"""
Compute ph-AFQMC nuclear gradients for a 2-hydrogen chain.

All parameters set to reproduce example/h2.py as a baseline.

NOTE: also need to set custom hardcoded eql rules:
    propagator_eq.n_ene_blocks = 5
    propagator_eq.n_sr_blocks = 10

Hmm...
Although the DF-RHF, RCCSD energy and gradients matches EXACTLY,
we don't get the same AFQMC results despite using the same seed...
The numbers are slightly off (up to AFQMC precision, NOT fp precision),
which means that the randomization was slightly different...
Perhaps it is because we touched something in the code, or it could
be because Jo hardcoded some values in the eql phase.

Either way reproducing the exact AFQMC values is not a must,
and it is good enough to know that we can reproduce this to a very
close number. Furthermore, we know that the eql and sampling
lengths are WAY too long, and in fact it only needed 2 superblocks
to reproduce the gradient to within 1sigma:

# Equilibration sweeps:
#   Iter        Block energy      Walltime
#     0      -1.088266453e+00     1.94e+00 
#     1      -1.119042277e+00     7.35e+00 
#     2      -1.119537354e+00     1.10e+01 
#     3      -1.119033217e+00     1.19e+01 
#     4      -1.120900512e+00     1.28e+01 
#     5      -1.120181561e+00     1.39e+01 
#     6      -1.120409250e+00     1.49e+01 
#     7      -1.118234992e+00     1.60e+01 
#     8      -1.119792461e+00     1.72e+01 
#     9      -1.119534135e+00     1.87e+01 
#    10      -1.118775964e+00     1.95e+01 
#
# Sampling sweeps:
#  Iter        Mean energy          Stochastic error       Mean observable       Walltime
     0       -1.119963330e+00                      -        0.000000000e+00           6.41e+01
     2       -1.119374134e+00        3.086505956e-04        0.000000000e+00           1.27e+02
^CAFQMC did not execute correctly.
4
(0,2) Couldnt find error estimate
(1,2) Couldnt find error estimate
Blocking analysis done
--------------- AFQMC Gradient ---------------

[[ 0.            0.           -0.1104526799]
 [ 0.            0.            0.1104526799]]
--------------- Error ---------------

[[0. 0. 0.]
 [0. 0. 0.]]
----------------------------------------------

Error is 0 because I signal killed it. But see how we already get 0.11045,
which is within 0.11063(19) in h2.out ... ok barely but it's sufficient.

Note how that in the example and here, our results deviate from RCCSD.
For 2 electrons, RCCSD is FCI, so should be exact for this basis+auxbasis.
RCCSD gets: 0.1043133599 which is 0.104ish
... 5.5% diff compared to 0.110ish from ph-AFQMC

The energies match better; RCCSD is -1.1207... vs -1.1194(2)
So we still have a bias, bc the value is over 3sigma away.

So is it that ph-AFQMC just has a bias for the gradient? 

For further reassurance: the FD ph-AFQMC tests show what the values SHOULD be
thanks to H2_FD_test/h2.py which turned out to be immensely helpful:
- same system, multi-point central difference
- the ph-AFQMC used here has 7x more walkers and 2.5x sampling time 17.5x
    - this brought energy to -1.11922(05) vs -1.11941(16)
    - and the gradient to     0.11084(06) vs  0.11063(19)
    - this makes sense, due to 17.5x more effective samples (?)
    - the decrease in error should be sqrt of this, which is 4.2x
    - actually observed decrease in error is about 3x... worth it?
- the best result from FD is 5p, got 0.11047(21), which is slightly lower
- but anyways AD is about the same accuracy as FD!

As a final note, these values are from no-rot gradients,
so there is no trial wfn response here!!

R = 1.8,dR = 0.1
FD obs_rhf_3p = -0.13878705864877272
FD obs_rhf_5p = -0.139577540597934
FD obs_rhf_7p = -0.13956432608562597
FD obs_afqmc_rhf_3p = -0.10967175782161775 +/- 0.00015587576124535265
FD obs_afqmc_rhf_5p = -0.11047018647004506 +/- 0.0002096036351377857
FD obs_afqmc_rhf_7p = -0.11047263479718679 +/- 0.00023894944042065596
"""

rH = 1.8
nH = 2
atom_symbols = np.array(["H"] * nH)
coords = np.zeros((nH, 3))
coords[:, 2] = np.arange(nH) * rH

atomstring = list(zip(atom_symbols, coords))
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
mf = scf.RHF(mol).density_fit()#auxbasis="def2-svp-jkfit") using autoaux...
# WARN: sto6g does not have predefined auxbasis, see pyscf.df.addons
# using sto3g and 321g's default, which should be reasonable
mf.kernel() # pyright: ignore[reportAttributeAccessIssue]

# each E-block step
n_walkers_per_proc = 100
nproc = 4
n_walkers_tot = nproc * n_walkers_per_proc
print(f"Total walkers: {n_walkers_tot}")
options = {
        "n_walkers": n_walkers_per_proc,
        "dt": 0.01,
        "n_prop_steps": 50, # sets sampling interval
        "n_ene_blocks": 8, # sets SR frequency
        "n_sr_blocks": 50, # sets preblock size

        "n_eql": 10,
        "n_blocks": 20, # sets n preblocks (along w nproc)

        "walker_type": "rhf",
        "trial": "rhf",

        "ad_mode": "reverse",
        "do_grad": True,
        "orbital_rotation": False, # note the norot!
        "do_sr": True,
        "seed": 948836,
}
sampling_interval = options['n_prop_steps'] * options['dt']
print(f"Energy sampling interval: {sampling_interval}")
sr_interval = options['n_ene_blocks'] * sampling_interval
print(f"Stochastic reconfig interval: {sr_interval}")
superblock_duration = sr_interval * options["n_sr_blocks"]
print(f"Superblock duration: {superblock_duration}")
# NOTE: we need a long enough superblock to allow response wrt stationary distribution changes due to perturbation in integrals
# similar to stop-gradient in SCF computations... eventually total response will not depend on the response of the initial state
print(f"Equilibration duration: {options['n_eql'] * superblock_duration}")
print(f"Sampling duration: {options['n_blocks'] * superblock_duration}")
print(f"Total nominal samples (sans blocking): {n_walkers_tot * options['n_ene_blocks'] * options["n_sr_blocks"] * options['n_blocks']}")
print(f"Total preblocked averages: {nproc * options['n_blocks']}")

input("Press enter to confirm and continue to AFQMC...")

print("Computing finite differences...")
grad_utils.FD_integrals(mf)            # Integral_der.npz
grad_utils.write_integrals_lowdins(mf) # FCIDUMP_chol, rhf.npz
# NOTE: rhf.npz was redundant work... LAO-dm was already done in FD_integrals
print("Done. Saved to disk. Proceed to run afqmcGrad...")
run_afqmc.run_afqmc(options=options,nproc=nproc) # bc do_grad==True, calls driver.afqmcGrad()
# takes FCIDUMP_chol and rhf.npz, outputs en_der_afqmc_*.npz
grad_utils.calculate_nuc_gradients(uhf=False) # uses Integral_der and en_der_afqmc_*
print(f"rH = {rH}")
mf_grad = mf.Gradients()
mf_grad.kernel()

mc = cc.CCSD(mf)
mc.kernel()
g = mc.nuc_grad_method().kernel()

