from functools import partial
from functools import reduce
import sys
print = partial(print, flush=True)

import numpy as np
import jax.numpy as jnp

import pyscf
from pyscf import gto, scf, cc
from ad_afqmc import run_afqmc, grad_utils


"""
Compute ph-AFQMC nuclear gradients for a hydrogen chain.
"""

rH = 1.8
nH = 2
BASIS = "sto-6g"
AUXBASIS_SPECFIC = None # None=default, alt: "def2-svp-jkfit"
# WARN: sto6g does not have predefined auxbasis, see pyscf.df.addons
# using sto3g and 321g's default, which should be reasonable

atom_symbols = np.array(["H"] * nH)
coords = np.zeros((nH, 3))
coords[:, 2] = np.arange(nH) * rH
atomstring = list(zip(atom_symbols, coords))
mol = gto.M(atom=atomstring, basis=BASIS, verbose=3, unit="bohr")
mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS_SPECFIC)
mf.kernel() # pyright: ignore[reportAttributeAccessIssue]

n_walkers_per_proc = int(sys.argv[1])
nproc = 4
n_walkers_tot = nproc * n_walkers_per_proc
print(f"Total walkers: {n_walkers_tot}")
options = {
        "dt": 0.01,
        "n_prop_steps": 50, # sets sampling interval
        "n_ene_blocks": 8, # sets SR frequency
        "n_sr_blocks":  20, # sets preblock size
        "n_blocks":     4, # sets n preblocks (along w nproc)

        "n_ene_per_sr_eql": 8,
        "n_sr_per_b_eql":   2,
        "n_eql":            10,

        "n_walkers": n_walkers_per_proc,
        "walker_type": "rhf",
        "trial": "rhf",

        "ad_mode": "reverse",
        "do_grad": True,
        "orbital_rotation": False,
        "do_sr": True,
        "seed": 250448,
}
sampling_interval = options['n_prop_steps'] * options['dt']
print(f"Energy sampling interval: {sampling_interval}")
sr_interval = options['n_ene_blocks'] * sampling_interval
print(f"Stochastic reconfig interval: {sr_interval}")
superblock_duration = sr_interval * options["n_sr_blocks"]
print(f"Superblock duration: {superblock_duration}")
# NOTE: we need a long enough superblock to allow response wrt stationary distribution changes due to perturbation in integrals
# similar to stop-gradient in SCF computations... eventually total response will not depend on the response of the initial state
print(f"Equilibration duration: {options['n_eql'] * options["n_ene_per_sr_eql"] * options["n_sr_per_b_eql"] * sampling_interval}")
print(f"Sampling duration: {options['n_blocks'] * superblock_duration}")
print(f"Total nominal samples (k): {n_walkers_tot * options['n_ene_blocks'] * options["n_sr_blocks"] * options['n_blocks'] / 1000}")
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
