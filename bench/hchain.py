import argparse
import json
from functools import partial
print = partial(print, flush=True)

import numpy as np
import jax.numpy as jnp

import pyscf
from pyscf import gto, scf, cc
from ad_afqmc import run_afqmc, grad_utils

"""
Compute ph-AFQMC nuclear gradients for a hydrogen chain. Run from command line.
"""

parser = argparse.ArgumentParser()
defaults = {
    # Physical system
    "rH": 1.8,
    "nH": 2,
    "basis": "sto-6g",
    "auxbasis": None,

    # Propagation
    "n_proc": 1,
    "n_walkers": 100,
    "do_sr": True,
    "seed": 250448,

    # Sampling
    "dt": 0.01,
    "n_prop_steps": 50,
    "n_ene_blocks": 8,
    "n_sr_blocks": 20,
    "n_blocks": 4,

    # Equilibration
    "n_ene_per_sr_eql": 8,
    "n_sr_per_b_eql": 2,
    "n_eql": 10,

    # Differentiation
    "orbital_rotation": False,
    "n_steps_per_ckpt": 0,
    "n_ene_per_ckpt": 1,
    "do_sr_ckpt": False,
}

g = parser.add_argument_group("Physical system")
g.add_argument("--rH", type=float)
g.add_argument("--nH", type=int)
g.add_argument("--basis", type=str)
g.add_argument("--auxbasis", type=str)

g = parser.add_argument_group("Propagation")
g.add_argument("--n_proc", type=int)
g.add_argument("--n_walkers", type=int)
g.add_argument("--do_sr", action=argparse.BooleanOptionalAction)
g.add_argument("--seed", type=int)

g = parser.add_argument_group("Sampling")
g.add_argument("--dt", type=float)
g.add_argument("--n_prop_steps", type=int)
g.add_argument("--n_ene_blocks", type=int)
g.add_argument("--n_sr_blocks", type=int)
g.add_argument("--n_blocks", type=int)

g = parser.add_argument_group("Equilibration")
g.add_argument("--n_ene_per_sr_eql", type=int)
g.add_argument("--n_sr_per_b_eql", type=int)
g.add_argument("--n_eql", type=int)

g = parser.add_argument_group("Differentiation")
g.add_argument("--orbital_rotation", action=argparse.BooleanOptionalAction)
g.add_argument("--n_steps_per_ckpt", type=int)
g.add_argument("--n_ene_per_ckpt", type=int)
g.add_argument("--do_sr_ckpt", action=argparse.BooleanOptionalAction)

parser.set_defaults(**defaults)
args = parser.parse_args()
options = { # explicit entries here not to be set by parser
    "walker_type": "rhf",
    "trial": "rhf",

    "ad_mode": "reverse",
    "do_grad": True,

    **vars(args),
}
with open("options.json", "w") as f:
    json.dump(options, f, indent=4)

rH = options.pop("rH")
nH = options.pop("nH")
BASIS = options.pop("basis")
AUXBASIS_SPECFIC = options.pop("auxbasis") # None=default, alt: "def2-svp-jkfit"
# WARN: sto6g does not have predefined auxbasis, see pyscf.df.addons
# using sto3g and 321g's default, which should be reasonable
nproc = options.pop("n_proc")

print("RHF computation")
atom_symbols = np.array(["H"] * nH)
coords = np.zeros((nH, 3))
coords[:, 2] = np.arange(nH) * rH
atomstring = list(zip(atom_symbols, coords))
mol = gto.M(atom=atomstring, basis=BASIS, verbose=3, unit="bohr")
mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS_SPECFIC)
mf.kernel(verbose=4) # pyright: ignore[reportAttributeAccessIssue]

print()
print("AFQMC computation")
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
n_walkers_tot = nproc * options['n_walkers']
print(f"Total walkers: {n_walkers_tot}")
print(f"Total nominal samples (k): {n_walkers_tot * options['n_ene_blocks'] * options["n_sr_blocks"] * options['n_blocks'] / 1000}")
print(f"Total preblocked averages: {nproc * options['n_blocks']}")

print()
print("Computing finite differences...")
grad_utils.FD_integrals(mf)            # Integral_der.npz
grad_utils.write_integrals_lowdins(mf) # FCIDUMP_chol, rhf.npz
# NOTE: rhf.npz was redundant work... LAO-dm was already done in FD_integrals
print("Done. Saved to disk. Proceed to run afqmcGrad...")
run_afqmc.run_afqmc(options=options,nproc=nproc) # bc do_grad==True, calls driver.afqmcGrad()
# takes FCIDUMP_chol and rhf.npz, outputs en_der_afqmc_*.npz
grad_utils.calculate_nuc_gradients(uhf=False) # uses Integral_der and en_der_afqmc_*

print()
print("Reference gradients")
mf_grad = mf.Gradients()
mf_grad.kernel()
print(f"rH = {rH}")
mc = cc.CCSD(mf)
mc.kernel()
g = mc.nuc_grad_method().kernel()
