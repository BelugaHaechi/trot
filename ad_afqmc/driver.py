import os
import time

import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import pickle
from copy import deepcopy
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax
import jax.numpy as jnp
from mpi4py import MPI

from ad_afqmc import linalg_utils, propagation, sampler, stat_utils, hamiltonian, wavefunctions

print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def afqmc(
    ham_data, # dict of values
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data, # dict of values
    observable, # dict of values
    options, # dict of stuff
    init_walkers=None
):
    """
    ph-AFQMC driver.

    Takes in data and functions generated from mpi_jax._prep_afqmc() (or by hand if needed).
    - has one more additional arg init_walkers for higher flexibility

    Process
    - observable operators for autodiff: loads 1b observable param matrix and the coupling constant
        - default is the H1 observable + 0... which yields hcore energy expectation value
            - its use is different from rdm_op, which is zeros_like
        - rdm_2_op, if requested by ad_mode == '2rdm', is built at O(N5) cost (!)
            - this is the full ERI with decomp error
            - bc this is huge, we only build it upon request
    - constant tensors for propagation: trial precontraction, halftime 1b propagator
        - calls ham methods, but not the AO-MO transform???
    - initialize prop_data dict containing walkers
        - calls propagator.init_prop_data
    - precontract trial with the observable operator
        - later I realized these are just fallback values...
        - trial_observable = tr(trial_rdm @ observable_op)
            - value of said observable with the trial 1rdm
        - if 2rdm mode, also computes the "2rdm" J - K/2
            - but this only works if the trial is RHF!
    - equilibration phase
        - makes "propagator_eq" as a deepcopy of propagator with modified values
        - hardcoded: 50 * (5 * 10) steps
        - entire phase is wrapped in a for loop (!) to be repeated 'neql' times
            - from options.neql, default 1
            - probably a dirty patch to have a longer eql phase
        - sampler.propagate_phaseless (_sr_block_scan)
            - MAIN CYCLE
            - standard, unmodified ph-AFQMC with SR (TODO: where is rot?)
        - then about 20 lines of MPI stuff that groups the block energy
        - orthonormalize walkers + global SR one more time
            - also done during main cycle: RO during ene_block, local SR during sr_block
        - update e_estimate as a 90/10 exponential moving average (EMA)
        - print the summary of the current block
    - sampling phase
        - default: 50 * (50 * 1) steps
        - prepares to collect energies and observables
            - ad_mode reverse actually does 1rdm, so reverse = 1rdm mode
            - ad_mode 2rdm does the 2rdm
        - chooses the propagator based on options: ad_mode, do_rot, do_sr
            - if ad_mode is not 2rdm, then has rot and sr options (4 combos)
            - else if ad_mode is 2rdm, then routes to "ad_1" (BAD NAMING)
                - I changed the name to "ad_rdm2", safe change
        - set up "prop_data_tangent", which is the "v" in jvp
            - only relevant for forward mode
            - goal is to make a thing shaped like prop_data but with all 0
                - probably a better way to do this using pytrees...
        - sets up similar zeros for rdm1 and rdm2 operators
            - but these are only initial values for storing observables
        - repeats propagate_phaseless_wrapper for n_blocks times
            - MAIN CYCLE
            - for loop, NOT scan !!!!! inefficient
            - huge elif for each ad_mode, interesting to analyze (see below)
            - the observable is computed here, has n_blocks jvp/vjp chained
        - then 40 lines of gathering values from MPI
        - orthonormalize walkers + global SR one more time
        - update e_estimate as a 90/10 exponential moving average (EMA)
        - print the summary of the block EVERY 10 blocks
            - stats_utils.blocking_analysis()
            - saves stuff to samples_raw.dat
    - final analysis
        - summarizes large deviations
        - 100 lines of final blocking_analysis()
            - yields final AFQMC energy
            - horrendous code
        - returns just E_AFQMC and stddev
    
    OK this is a pain to figure out. I'd just assume it works

    My job is to optimize the MAIN CYCLE code to use less memory.

    - ask what orbital rotation is for
    - sampler code (especially how the observable is treated)
    """
    init = time.time()
    seed = options["seed"]
    neql = options["n_eql"]

    if observable is not None:
        observable_op = jnp.array(observable[0])
        observable_constant = observable[1]
    else:
        observable_op = jnp.array(ham_data["h1"])
        observable_constant = 0.0

    rdm_op = 0.0 * jnp.array(ham_data["h1"])  # for reverse mode
    rdm_2_op = None
    if options["ad_mode"] == "2rdm":
        nchol = ham.nchol
        norb = ham.norb
        eri_full = np.einsum(
            "gj,gl->jl",
            ham_data["chol"].reshape(nchol, -1),
            ham_data["chol"].reshape(nchol, -1),
        )
        # # convert to symmetry 4
        # # very slow, but only done once
        # npair = norb * (norb + 1) // 2
        # eri = np.zeros((npair, npair))
        # for ij in range(npair):
        #     i = int(np.floor(np.sqrt(2 * ij + 0.25) - 0.5))
        #     j = ij - i * (i + 1) // 2
        #     ij_full = i * norb + j
        #     eri[ij, ij] = eri_full[ij_full, ij_full]
        #     for kl in range(ij):
        #         k = int(np.floor(np.sqrt(2 * kl + 0.25) - 0.5))
        #         l = kl - k * (k + 1) // 2
        #         kl_full = k * norb + l
        #         eri[ij, kl] = eri_full[ij_full, kl_full]
        #         eri[kl, ij] = eri[ij, kl]
        # ham_data["chol"] = linalg_utils.modified_cholesky(eri, ham.norb, ham.nchol)
        rdm_2_op = jnp.array(eri_full).reshape(norb, norb, norb, norb)

    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)   
    prop_data = propagator.init_prop_data(trial, wave_data, ham, ham_data, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    prop_data["key"] = jax.random.PRNGKey(seed + rank)
    trial_rdm1 = trial.get_rdm1(wave_data)
    trial_observable = np.sum(trial_rdm1 * observable_op)
    trial_rdm2 = None
    if options["ad_mode"] == "2rdm":
        trial_rdm1_spatial = trial_rdm1[0] + trial_rdm1[1]
        trial_rdm2 = (
            np.einsum("ij,kl->ijkl", trial_rdm1_spatial, trial_rdm1_spatial)
            - np.einsum("ij,kl->iklj", trial_rdm1_spatial, trial_rdm1_spatial) / 2
        )
        trial_rdm2 = trial_rdm2.reshape(ham.norb**2, ham.norb**2)
        trial_rdm2 = jnp.array(trial_rdm2)

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("# Equilibration sweeps:")
        print("#   Iter        Block energy      Walltime")
        n = 0
        print(f"# {n:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2e} ")
    comm.Barrier()

    # if isinstance(propagator, propagation.propagator_cpmc):
    #     propagator_eq = propagation.propagator_cpmc(
    #         propagator.dt,
    #         n_prop_steps=50,
    #         n_ene_blocks=5,
    #         n_sr_blocks=10,
    #         n_walkers=options["n_walkers"],
    #     )
    # elif isinstance(propagator, propagation.propagator_cpmc_continuous):
    #     propagator_eq = propagation.propagator_cpmc_continuous(
    #         propagator.dt,
    #         n_prop_steps=50,
    #         n_ene_blocks=5,
    #         n_sr_blocks=10,
    #         n_walkers=options["n_walkers"],
    #     )
    # elif options["walker_type"] == "rhf":
    #     propagator_eq = propagation.propagator(
    #         propagator.dt,
    #         n_prop_steps=50,
    #         n_ene_blocks=5,
    #         n_sr_blocks=10,
    #         n_walkers=options["n_walkers"],
    #     )
    # elif options["walker_type"] == "uhf":
    #     propagator_eq = propagation.propagator_uhf(
    #         propagator.dt,
    #         n_prop_steps=50,
    #         n_ene_blocks=5,
    #         n_sr_blocks=10,
    #         n_walkers=options["n_walkers"],
    #     )
    propagator_eq = deepcopy(propagator)
    propagator_eq.n_prop_steps = 50 # WARN: hardcoded
    propagator_eq.n_ene_blocks = 5
    propagator_eq.n_sr_blocks = 10
    propagator_eq.n_walkers = options["n_walkers"] # redundant?

    for n in range(1, neql + 1):
        block_energy_n, prop_data = sampler.propagate_phaseless(
            ham, ham_data, propagator_eq, prop_data, trial, wave_data
        )
        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_weighted_energy_n = np.array(
            [block_energy_n * block_weight_n], dtype="float32"
        )
        total_block_energy_n = np.zeros(1, dtype="float32")
        total_block_weight_n = np.zeros(1, dtype="float32")
        comm.Reduce(
            [block_weighted_energy_n, MPI.FLOAT],
            [total_block_energy_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        comm.Reduce(
            [block_weight_n, MPI.FLOAT],
            [total_block_weight_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        if rank == 0:
            block_weight_n = total_block_weight_n
            block_energy_n = total_block_energy_n / total_block_weight_n
        comm.Bcast(block_weight_n, root=0)
        comm.Bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)
        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = (
            0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n[0]
        )

        comm.Barrier()
        if rank == 0:
            print(
                f"# {n:5d}      {block_energy_n[0]:.9e}     {time.time() - init:.2e} ",
                flush=True,
            )
        comm.Barrier()

    local_large_deviations = np.array(0)

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("#\n# Sampling sweeps:")
        if options["ad_mode"] is None:
            print("#  Iter        Mean energy          Stochastic error       Walltime")
        else:
            print(
                "#  Iter        Mean energy          Stochastic error       Mean observable       Walltime"
            )
    comm.Barrier()

    global_block_weights = None
    global_block_energies = None
    global_block_observables = None
    global_block_rdm1s = None
    global_block_rdm2s = None
    if rank == 0:
        global_block_weights = np.zeros(size * propagator.n_blocks)
        global_block_energies = np.zeros(size * propagator.n_blocks)
        global_block_observables = np.zeros(size * propagator.n_blocks)
        if options["ad_mode"] == "reverse":
            global_block_rdm1s = np.zeros(
                (size * propagator.n_blocks, *(ham_data["h1"].shape))
            )
        elif options["ad_mode"] == "2rdm":
            global_block_rdm2s = np.zeros(
                (size * propagator.n_blocks, *(rdm_2_op.shape))
            )

    if options["orbital_rotation"] == False and options["do_sr"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_nosr_norot(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    elif options["orbital_rotation"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_norot(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    elif options["do_sr"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_nosr(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    else:
        propagate_phaseless_wrapper = lambda x, y, z: sampler.propagate_phaseless_ad(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )

    if options["ad_mode"] == "2rdm":
        propagate_phaseless_wrapper = lambda x, y, z: sampler.propagate_phaseless_ad_rdm2(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )

    prop_data_tangent = {}
    for x in prop_data:
        if isinstance(prop_data[x], list):
            prop_data_tangent[x] = [np.zeros_like(y) for y in prop_data[x]]
        elif prop_data[x].dtype == "uint32":
            prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=jax.dtypes.float0)
        else:
            prop_data_tangent[x] = np.zeros_like(prop_data[x])
    block_rdm1_n = np.zeros_like(ham_data["h1"])
    block_rdm2_n = None
    if options["ad_mode"] == "2rdm":
        block_rdm2_n = np.zeros_like(rdm_2_op)
    block_observable_n = 0.0

    for n in range(propagator.n_blocks):
        if options["ad_mode"] == "forward":
            coupling = 0.0
            block_energy_n, block_observable_n, prop_data = jax.jvp(
                propagate_phaseless_wrapper,
                (coupling, observable_op, prop_data),
                (1.0, 0.0 * observable_op, prop_data_tangent),
                has_aux=True,
            )
            if np.isnan(block_observable_n) or np.isinf(block_observable_n):
                block_observable_n = trial_observable
                local_large_deviations += 1
        elif options["ad_mode"] == "reverse":
            coupling = 1.0
            block_energy_n, block_vjp_fun, prop_data = jax.vjp(
                propagate_phaseless_wrapper, coupling, rdm_op, prop_data, has_aux=True
            )
            block_rdm1_n = block_vjp_fun(1.0)[1]
            block_observable_n = np.sum(block_rdm1_n * observable_op)
            if np.isnan(block_observable_n) or np.isinf(block_observable_n):
                block_observable_n = trial_observable
                block_rdm1_n = trial_rdm1
                local_large_deviations += 1
        elif options["ad_mode"] == "2rdm":
            coupling = 1.0
            block_energy_n, block_vjp_fun, prop_data = jax.vjp(
                propagate_phaseless_wrapper, coupling, rdm_2_op, prop_data, has_aux=True
            )
            block_rdm2_n = block_vjp_fun(1.0)[1]
            block_observable_n = trial_observable
            # if np.isnan(block_observable_n) or np.isinf(block_observable_n) :
            if np.isnan(np.linalg.norm(block_rdm2_n)) or np.isinf(
                np.linalg.norm(block_rdm2_n)
            ):
                block_observable_n = trial_observable
                block_rdm2_n = trial_rdm2
                local_large_deviations += 1
        else:
            block_energy_n, prop_data = sampler.propagate_phaseless(
                ham, ham_data, propagator, prop_data, trial, wave_data
            )
            block_observable_n = 0.0

        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_observable_n = np.array(
            [block_observable_n + observable_constant], dtype="float32"
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_rdm1_n = np.array(block_rdm1_n, dtype="float32")
        if options["ad_mode"] == "2rdm":
            block_rdm2_n = np.array(block_rdm2_n, dtype="float32")

        gather_weights = None
        gather_energies = None
        gather_observables = None
        gather_rdm1s = None
        gather_rdm2s = None
        if rank == 0:
            gather_weights = np.zeros(size, dtype="float32")
            gather_energies = np.zeros(size, dtype="float32")
            gather_observables = np.zeros(size, dtype="float32")
            if options["ad_mode"] == "reverse":
                gather_rdm1s = np.zeros(
                    (size, *(ham_data["h1"].shape)), dtype="float32"
                )
            elif options["ad_mode"] == "2rdm":
                gather_rdm2s = np.zeros((size, *(rdm_2_op.shape)), dtype="float32")

        comm.Gather(block_weight_n, gather_weights, root=0)
        comm.Gather(block_energy_n, gather_energies, root=0)
        comm.Gather(block_observable_n, gather_observables, root=0)
        if options["ad_mode"] == "reverse":
            comm.Gather(block_rdm1_n, gather_rdm1s, root=0)
        elif options["ad_mode"] == "2rdm":
            comm.Gather(block_rdm2_n, gather_rdm2s, root=0)
        block_energy_n = 0.0
        if rank == 0:
            global_block_weights[n * size : (n + 1) * size] = gather_weights
            global_block_energies[n * size : (n + 1) * size] = gather_energies
            global_block_observables[n * size : (n + 1) * size] = gather_observables
            if options["ad_mode"] == "reverse":
                global_block_rdm1s[n * size : (n + 1) * size] = gather_rdm1s
            elif options["ad_mode"] == "2rdm":
                global_block_rdm2s[n * size : (n + 1) * size] = gather_rdm2s
            block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
                gather_weights
            )

        block_energy_n = comm.bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)

        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data, f)

        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        if n % (max(propagator.n_blocks // 10, 1)) == 0:
            comm.Barrier()
            if rank == 0:
                e_afqmc, energy_error = stat_utils.blocking_analysis(
                    global_block_weights[: (n + 1) * size],
                    global_block_energies[: (n + 1) * size],
                    neql=0,
                )
                obs_afqmc, _ = stat_utils.blocking_analysis(
                    global_block_weights[: (n + 1) * size],
                    global_block_observables[: (n + 1) * size],
                    neql=0,
                )
                if energy_error is not None:
                    if options["ad_mode"] is None:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {time.time() - init:.2e} ",
                            flush=True,
                        )
                    else:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                            flush=True,
                        )
                else:
                    if options["ad_mode"] is None:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}                -              {time.time() - init:.2e} ",
                            flush=True,
                        )
                    else:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}                -              {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                            flush=True,
                        )
                np.savetxt(
                    "samples_raw.dat",
                    np.stack(
                        (
                            global_block_weights[: (n + 1) * size],
                            global_block_energies[: (n + 1) * size],
                            global_block_observables[: (n + 1) * size],
                        )
                    ).T,
                )
            comm.Barrier()

    global_large_deviations = np.array(0)
    comm.Reduce(
        [local_large_deviations, MPI.INT],
        [global_large_deviations, MPI.INT],
        op=MPI.SUM,
        root=0,
    )
    comm.Barrier()
    if rank == 0:
        print(f"#\n# Number of large deviations: {global_large_deviations}", flush=True)

    comm.Barrier()
    e_afqmc, e_err_afqmc = None, None
    if rank == 0:
        np.savetxt(
            "samples_raw.dat",
            np.stack(
                (global_block_weights, global_block_energies, global_block_observables)
            ).T,
        )
        if options["ad_mode"] is not None and options["ad_mode"] != "2rdm":
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                2,
            )
        else:
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                1,
            )

        print(
            f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
        np.savetxt("samples.dat", samples_clean)
        global_block_weights = samples_clean[:, 0]
        global_block_energies = samples_clean[:, 1]
        global_block_observables = samples_clean[:, 2]
        if options["ad_mode"] == "reverse":
            global_block_rdm1s = global_block_rdm1s[idx]
        elif options["ad_mode"] == "2rdm":
            global_block_rdm2s = global_block_rdm2s[idx]

        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            global_block_weights, global_block_energies, neql=0, printQ=True
        )
        if e_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
            sig_err = np.around(
                np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_e = np.around(e_afqmc, sig_dec)
            print(f"AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
        elif e_afqmc is not None:
            print(f"AFQMC energy: {e_afqmc}\n", flush=True)
            e_err_afqmc = 0.0

        if options["ad_mode"] is not None:
            obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                global_block_weights, global_block_observables, neql=0, printQ=True
            )
            if err_afqmc is not None:
                sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
                sig_err = np.around(
                    np.round(err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
                )
                sig_obs = np.around(obs_afqmc, sig_dec)
                print(
                    f"AFQMC observable: {sig_obs:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n"
                )
            elif obs_afqmc is not None:
                print(f"AFQMC observable: {obs_afqmc}\n", flush=True)
            if options["ad_mode"] == "reverse":
                # avg_rdm1 = np.einsum('i,i...->...', global_block_weights, global_block_rdm1s) / np.sum(global_block_weights)
                norms_rdm1 = np.array(list(map(np.linalg.norm, global_block_rdm1s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm1)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm1s = global_block_rdm1s[idx]
                avg_rdm1 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm1s
                ) / np.sum(global_block_weights)
                errors_rdm1 = np.array(
                    list(map(np.linalg.norm, global_block_rdm1s - avg_rdm1))
                ) / np.linalg.norm(avg_rdm1)
                print(f"# RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm1, neql=0, printQ=True
                )
                np.savez("rdm1_afqmc.npz", rdm1=avg_rdm1)
            elif options["ad_mode"] == "2rdm":
                norms_rdm2 = np.array(list(map(np.linalg.norm, global_block_rdm2s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm2)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm2s = global_block_rdm2s[idx]
                avg_rdm2 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm2s
                ) / np.sum(global_block_weights)
                errors_rdm2 = np.array(
                    list(map(np.linalg.norm, global_block_rdm2s - avg_rdm2))
                ) / np.linalg.norm(avg_rdm2)
                print(f"# 2RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm2, neql=0, printQ=True
                )
                np.savez(
                    "rdm2_afqmc.npz",
                    rdm2=2 * avg_rdm2.reshape(ham.norb, ham.norb, ham.norb, ham.norb),
                )

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()
    return e_afqmc, e_err_afqmc



def afqmcGrad(
    ham_data, # dict of values
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data, # dict of values
    observable, # dict of values
    options, # dict of stuff
    init_walkers=None
):
    """
    ph-AFQMC with nuclear gradients workflow.

    Input: CRITIAL: all are in LowdinAO basis, RHF level
    - ham_data
        - h1, raw from FCIDUMP (NO v0 modification)
        - chol, raw from FCIDUMP
        - h0, constant made in pyscf
    - wave_data: mo_coeff from rhf.npz

    NOTE: without ham.rot_orbs(), all quantities will stay in the basis they came from
        - quantities: h1, chol, trial mo_coeff
        - this basis is transformed beforehand in FCIDUMP, from AO to LAO basis
        - hence these quantities remain in LAO basis, as long as no ham.rot_orbs() is called!!!
    
    OK in fact, rot_orbs is never called in the driver, which is good
    - it DOES appear ONLY in ad, ad_rdm2, ad_nosr
    - DOES NOT happen in any nucgrad_* nor ad_norot

    In nucgrad, it seems that we would like to stay in the LAO basis, so ok good
    """
    from ad_afqmc import grad_utils
    init = time.time()
    seed = options["seed"]

    # load observable operators
    if observable is not None:
        observable_op = jnp.array(observable[0])
        observable_constant = observable[1]
    else:
        observable_op = jnp.array(ham_data["h1"])
        observable_constant = 0.0
    # copies of h1 and chol... LAO basis
    # later be used as INPUT for vjp for nucgrad, so we can take derivs wrt to them
    norb = ham.norb
    rdm_op = jnp.array(ham_data["h1"]).copy()  
    rdm_2_op = ham_data["chol"].reshape((-1,norb,norb)).copy()
    # if the target observable is the full 2rdm, rebuild the full ERI tensor
    # from the CDERIs (which represent the 2rdm, but isn't the full thing)
    if options["ad_mode"] == "2rdm":
        nchol = ham.nchol
        norb = ham.norb
        eri_full = np.einsum(
            "gj,gl->jl",
            ham_data["chol"].reshape(nchol, -1),
            ham_data["chol"].reshape(nchol, -1),
        )
        # # convert to symmetry 4
        # # very slow, but only done once
        # npair = norb * (norb + 1) // 2
        # eri = np.zeros((npair, npair))
        # for ij in range(npair):
        #     i = int(np.floor(np.sqrt(2 * ij + 0.25) - 0.5))
        #     j = ij - i * (i + 1) // 2
        #     ij_full = i * norb + j
        #     eri[ij, ij] = eri_full[ij_full, ij_full]
        #     for kl in range(ij):
        #         k = int(np.floor(np.sqrt(2 * kl + 0.25) - 0.5))
        #         l = kl - k * (k + 1) // 2
        #         kl_full = k * norb + l
        #         eri[ij, kl] = eri_full[ij_full, kl_full]
        #         eri[kl, ij] = eri[ij, kl]
        # ham_data["chol"] = linalg_utils.modified_cholesky(eri, ham.norb, ham.nchol)
        rdm_2_op = jnp.array(eri_full).reshape(norb, norb, norb, norb)

    if rank==0: print("Precomputing contractions and propagators...")
    # build trial-precontracted tensors for energy eval... LAO basis
    ham_data = ham.rot_ham(ham_data, wave_data)
    # build constant exp_1e propagator, and mean-shift constants... LAO basis
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)   
    if rank==0: print("Done. Initializing walkers...")
    # init walker population to be n_walker copies of the trial RHF det
    prop_data = propagator.init_prop_data(trial, wave_data, ham, ham_data, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    prop_data["key"] = jax.random.PRNGKey(seed + rank)

    # observable, RDM1, RDM2 of the trial state, as constants
    # serves as a FALLBACK for the observable value when "local large deviations" happen
    # see the main loop code below for references. these values are unused until much later
    # dlai: I've changed the varnames for better readability
    trial_rdm1 = trial.get_rdm1(wave_data)
    FALLBACK_TRIAL_STATE_RDM1 = trial_rdm1                                  # ad_mode==reverse
    FALLBACK_TRIAL_STATE_OBSERVABLE = np.sum(trial_rdm1 * observable_op)    # forward,reverse,2rdm
    FALLBACK_TRIAL_STATE_RDM2 = None
    if options["ad_mode"] == "2rdm":
        trial_rdm1_spatial = trial_rdm1[0] + trial_rdm1[1]
        FALLBACK_TRIAL_STATE_RDM2 = (
            np.einsum("ij,kl->ijkl", trial_rdm1_spatial, trial_rdm1_spatial)
            - np.einsum("ij,kl->iklj", trial_rdm1_spatial, trial_rdm1_spatial) / 2
        )
        FALLBACK_TRIAL_STATE_RDM2 = FALLBACK_TRIAL_STATE_RDM2.reshape(ham.norb**2, ham.norb**2)
        FALLBACK_TRIAL_STATE_RDM2 = jnp.array(FALLBACK_TRIAL_STATE_RDM2)


    # EQUILIBRATION PHASE
    # use the same propagator as the sampling phase
    # dlai: make these adjustable options
    propagator_eq = deepcopy(propagator)
    propagator_eq.n_ene_blocks = options["n_ene_per_sr_eql"]
    propagator_eq.n_sr_blocks = options["n_sr_per_b_eql"]
    neql = options["n_eql"]

    # address annoying bug where this key was added to the dict only after iteration 1
    # someone forgot to initialize this! for some bad reason it's a float
    prop_data["n_killed_walkers"] = jnp.array(0, dtype=jnp.float64)

    # Compile equilibration
    comm.Barrier()
    t0 = time.perf_counter()

    propagate_eq_fn = sampler.propagate_phaseless.lower(
        ham, ham_data, propagator_eq, prop_data, trial, wave_data
    ).compile()

    comm.Barrier()
    if rank == 0:
        print(f"# Equilibration compilation walltime: {time.perf_counter() - t0:.3f} s")

    # Start timing iterations after compilation
    init_time = time.time() - init
    if rank == 0:
        print("# Equilibration sweeps:")
        print("#   Iter        Block energy      Walltime")
        print(f"# {0:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2f} ")
    comm.Barrier()

    for n in range(1, neql + 1):
        block_energy_n, prop_data = propagate_eq_fn(ham_data, prop_data, wave_data)

        # preprocessing of the data: wrap them into len-1 arrays, for comm.Reduce

        # block energy (average of n_walker/nproc samples of n_ene*n_sr observations every 50 steps)
        block_energy_n = np.array([block_energy_n], dtype="float32")
        # WARN: block weight is taken to be simply total weight at latest timestep!
        # inconsistent with the choice above. should be the big average like the energy
        # but bc of SR, the weights are always about the same so we get away with it
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        # "weighted_energy" which is the energy numerator <T|H|AFQMC>
        # yeah right a weightless formulation would make things a lot easier...
        # oh but we're doing phaseless. have to stick with the cumbersome weights...
        block_weighted_energy_n = np.array(
            [block_energy_n * block_weight_n], dtype="float32"
        )
        # mpi-sum the values over all ranks
        total_block_weighted_energy_n = np.zeros(1, dtype="float32")
        total_block_weight_n = np.zeros(1, dtype="float32")
        comm.Reduce(
            [block_weighted_energy_n, MPI.FLOAT],
            [total_block_weighted_energy_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        comm.Reduce(
            [block_weight_n, MPI.FLOAT],
            [total_block_weight_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        # compute the global weight and energy, then share with all ranks
        if rank == 0:
            block_weight_n = total_block_weight_n
            block_energy_n = total_block_weighted_energy_n / total_block_weight_n
        comm.Bcast(block_weight_n, root=0)
        comm.Bcast(block_energy_n, root=0)
        # do RO and global SR and update hat(E0)
        prop_data = propagator.orthonormalize_walkers(prop_data)
        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = (
            0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n[0]
        )
        # print results every superblock
        comm.Barrier()
        if rank == 0:
            print(
                f"# {n:5d}      {block_energy_n[0]:.9e}     {time.time() - init:.2f} ",
                flush=True,
            )
        comm.Barrier()

    # SAMPLING PHASE
    # NOTE: FROM THIS POINT ON, WE ASSUME THAT SUBSEQUENT PROPAGATION DOES
    # NOT STATISTICALLY IMPROVE THE EXPECTATION VALUE OF ANY OBSERVABLES.
    # WE ASSUME THAT EACH OBSERVATION FOLLOWS A WEAKLY STATIONARY PROCESS.

    # prepare storage for preblocked observable averages, for blocking analysis
    # length n_blocks * nproc, each element same shape as observable
    # for same superblock, the nproc block averages are independent up to SR
    # between different superblocks, the block averages are autocorrelated
    global_block_weights: np.ndarray
    global_block_energies: np.ndarray
    global_block_observables: np.ndarray
    global_block_rdm1s: np.ndarray
    global_block_rdm2s: np.ndarray
    if rank == 0:
        global_block_weights = np.empty(size * propagator.n_blocks)
        global_block_energies = np.empty(size * propagator.n_blocks)
        global_block_observables = np.empty(size * propagator.n_blocks)
        if options["ad_mode"] == "reverse": # "reverse" is really 1rdm mode
            global_block_rdm1s = np.empty(
                (size * propagator.n_blocks, *(rdm_op.shape))
            )
        elif options["ad_mode"] == "2rdm": # reverse but want full 2rdm
            global_block_rdm2s = np.empty(
                (size * propagator.n_blocks, *(rdm_2_op.shape))
            )

    # select propagation scheme based on options, then build into wrapper
    # dlai: I made this a LOT more concise
    if options["ad_mode"] == "2rdm":
        superblock_propagator = sampler.propagate_phaseless_ad_rdm2
    else:
        match (options["do_grad"], options["orbital_rotation"], options["do_sr"]):
            case (True, False, False):
                superblock_propagator = sampler.propagate_phaseless_nucgrad_norot_nosr
            case (True, False, True):
                superblock_propagator = sampler.propagate_phaseless_nucgrad_norot
            case (True, True, False):
                superblock_propagator = sampler.propagate_phaseless_nucgrad_rot_nosr
            case (True, True, True):
                superblock_propagator = sampler.propagate_phaseless_nucgrad
            case (False, False, False):
                superblock_propagator = sampler.propagate_phaseless_ad_nosr_norot
            case (False, False, True):
                superblock_propagator = sampler.propagate_phaseless_ad_norot
            case (False, True, False):
                superblock_propagator = sampler.propagate_phaseless_ad_nosr
            case (False, True, True):
                superblock_propagator = sampler.propagate_phaseless_ad

    # print('Using propagator:', getattr(superblock_propagator, "__name__", "unknown"))
    # remaining args: coupling, observable_op, prop_data    for ad
    #                 rdm1op,   rdm2op,        prop_data    for nucgrad
    # WARN: functools.partial does not work here! (I tried)
    # the reason is that jit-wrapped functions can't accept non-pytree args like a partial'd function
    # and jax.value_and_grad calls vjp and jit internally, so we get a nasty TypeError
    def propagate_phaseless_wrapper(obs_part_1, obs_part_2, prop_data):
        return superblock_propagator(ham, ham_data, obs_part_1, obs_part_2, propagator, prop_data, trial, wave_data)

    # construct the zero tangent components for fwd-AD (jvp)
    def make_zero_tangent(x):
        if jnp.issubdtype(x.dtype, jnp.integer):
            return jnp.zeros(x.shape, dtype=jax.dtypes.float0)
        return jnp.zeros_like(x)
    prop_data_tangent = jax.tree.map(make_zero_tangent, prop_data)
    observable_op_tangent = jnp.zeros_like(observable_op)


    # MAIN CYCLE
    block_rdm1_n = np.zeros_like(rdm_op)
    block_rdm2_n = np.zeros_like(rdm_2_op)
    block_observable_n = 0.0
    local_large_deviations = np.array(0)

    # compile before timing each superblock
    if options["do_grad"]:
        value_and_grad_fn = jax.jit(
            jax.value_and_grad(
                propagate_phaseless_wrapper,
                argnums=(0, 1),
                has_aux=True,
            )
        )

        # dlai: print jaxpr and residuals to file for reference
        if rank == 0:
            with open("jaxpr.txt", "w") as f:
                f.write(
                    jax.make_jaxpr(value_and_grad_fn)(rdm_op, rdm_2_op, prop_data)
                    .pretty_print(use_color=False)
                )

            import contextlib
            from jax.ad_checkpoint import print_saved_residuals

            def func(rdm_op, rdm_2_op):
                return propagate_phaseless_wrapper(rdm_op, rdm_2_op, prop_data)

            with open("saved_residuals.txt", "w") as fout:
                with contextlib.redirect_stdout(fout):
                    print_saved_residuals(func, rdm_op, rdm_2_op)

        # Explicit compilation on all MPI ranks
        comm.Barrier()
        t0 = time.perf_counter()

        value_and_grad_fn = value_and_grad_fn.lower(rdm_op, rdm_2_op, prop_data).compile()

        comm.Barrier()
        if rank == 0:
            print('#')
            print(f"# Sampling compilation walltime: {time.perf_counter() - t0:.3f} s")

    # Iteration 0 time now includes compilation
    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("# Sampling sweeps:")
        if options["ad_mode"] is None:
            print("#  Iter      Mean energy        Stochastic error     Walltime")
        else:
            print("#  Iter      Mean energy        Stochastic error     Mean observable     Walltime")
            print(f"     0     {time.time() - init:>70.2f}")
    comm.Barrier()

    for n in range(propagator.n_blocks):
        # choose mode...
        # nuclear gradient mode by Jo
        # need both d(hcore) and d(chol), aka d(rdm_op) and d(rdm_2_op)
        # also need trial mo_coeff to depend on hcore and chol!! tricky part
        # - this part is dealt with within the wrapper
        if options["do_grad"]:
            # combined forward and backward pass
            # dlai: value_and_grad is more explicit than vjp
            value, gradient = value_and_grad_fn(rdm_op, rdm_2_op, prop_data)

            # save values to disk
            block_energy_n, prop_data = value
            block_rdm1_n, block_rdm2_n = gradient
            grad_utils.append_to_array(f"en_der_afqmc_{comm.rank}.npz",
                block_rdm1_n, block_rdm2_n, jnp.sum(prop_data["weights"]))

        # forward mode
        # response formalism to get the expectation value of a SINGLE observable
        # 1. need to modify the hamiltonian's H1 term to add cA
        #   - where A is the 1b observable operator and c is set to 0
        # 2. set the tangent to be 1 for c and 0 for everything else
        elif options["ad_mode"] == "forward":
            coupling = 0.0
            block_energy_n, block_observable_n, prop_data = jax.jvp(
                propagate_phaseless_wrapper,
                (coupling, observable_op, prop_data),
                (1.0, observable_op_tangent, prop_data_tangent),
                has_aux=True,
            )
            # if deriv blows up, fallback to the trial's value...
            if np.isnan(block_observable_n) or np.isinf(block_observable_n):
                block_observable_n = FALLBACK_TRIAL_STATE_OBSERVABLE
                local_large_deviations += 1
        
        # reverse mode for 1rdm
        # see eq. 21 in the response paper
        # modify H1 operator to H1 + 1 * [0]ij = H1 so fwd pass unchanged
        # then rev pass would diff the entire [0]ij, which yields <a_ij> as needed
        # this also works, another easier impl would be "direct replace" as done in 2rdm
        # that is, just set H1 to rdm_op to make it depend on it
        # NOTE: in afqmcGrad rdm_op is defined to be hcore instead of zeros-like-hcore
        # so I had to explicitly introduce rdm_op_zero to make it work again
        elif options["ad_mode"] == "reverse":
            coupling = 1.0
            rdm_op_zero = jnp.zeros_like(rdm_op)
            block_energy_n, block_vjp_fun, prop_data = jax.vjp(
                propagate_phaseless_wrapper, coupling, rdm_op_zero, prop_data, has_aux=True
            )
            block_rdm1_n = block_vjp_fun(1.0)[1]
            block_observable_n = np.sum(block_rdm1_n * observable_op)
            if np.isnan(block_observable_n) or np.isinf(block_observable_n):
                block_observable_n = FALLBACK_TRIAL_STATE_OBSERVABLE
                block_rdm1_n = FALLBACK_TRIAL_STATE_RDM1
                local_large_deviations += 1

        # reverse mode for 2rdm (4index)
        # see eq. 21 in the response paper
        # actual impl: overrides chol with... chol but this time it is an input arg!
        # what really happens:
        # 1. at the beginning, we get chol from FCIDUMP
        # 2. we convert chol to eri (3idx to 4idx), call it rdm_2_op
        # 3. feed the eri into the ad_rdm2 wrapper as INPUT
        # 4. in the wrapper, convert eri back to chol (LOL)
        # 5. run a non-pyscf SCF with this chol as eri and the trial as dm0
        # 6. the SCF will converge like almost instantly, use this as new trial
        # 7. but now voila, this new trial is almost the old trial... except that it
        #    is now a function of the eri (!) AND also autodiffable thanks to JAX!
        # 8. now convert h1 and chol from AO to this new MO basis from the new trial
        # 9. proceed with normal ph-AFQMC, but now the entire ph-AFQMC depends on ERI!
        # 10. use vjp to get the ERI derivatives, neat!
        elif options["ad_mode"] == "2rdm": # reverse 2rdm
            coupling = 1.0 # actually unused!
            block_energy_n, block_vjp_fun, prop_data = jax.vjp(
                propagate_phaseless_wrapper, coupling, rdm_2_op, prop_data, has_aux=True
            )
            block_rdm2_n = block_vjp_fun(1.0)[1]
            block_observable_n = FALLBACK_TRIAL_STATE_OBSERVABLE
            # if np.isnan(block_observable_n) or np.isinf(block_observable_n) :
            if np.isnan(np.linalg.norm(block_rdm2_n)) or np.isinf(
                np.linalg.norm(block_rdm2_n)
            ):
                block_observable_n = FALLBACK_TRIAL_STATE_OBSERVABLE
                block_rdm2_n = FALLBACK_TRIAL_STATE_RDM2
                local_large_deviations += 1

        # normal ph-AFQMC mode. fallback if no AD requested.
        else:
            block_energy_n, prop_data = sampler.propagate_phaseless(
                ham, ham_data, propagator, prop_data, trial, wave_data
            )
            block_observable_n = 0.0

        # again, prepare storage for preblocked observable averages, for blocking analysis
        # same as eql phase, except there is also observables now
        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_observable_n = np.array(
            block_observable_n + observable_constant, dtype="float32"
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_rdm1_n = np.array(block_rdm1_n, dtype="float32")
        if options["ad_mode"] == "2rdm":
            block_rdm2_n = np.array(block_rdm2_n, dtype="float32")

        gather_weights: np.ndarray
        gather_energies: np.ndarray
        gather_observables: np.ndarray
        gather_rdm1s: np.ndarray
        gather_rdm2s: np.ndarray
        if rank == 0:
            gather_weights = np.empty(size, dtype=np.float32)
            gather_energies = np.empty(size, dtype=np.float32)
            gather_observables = np.empty(size, dtype=np.float32)
            if options["ad_mode"] == "reverse":
                gather_rdm1s = np.empty(
                    (size, *ham_data["h1"].shape),
                    dtype=np.float32,
                )
            elif options["ad_mode"] == "2rdm":
                gather_rdm2s = np.empty(
                    (size, *rdm_2_op.shape),
                    dtype=np.float32,
                )

            comm.Gather(block_weight_n, gather_weights, root=0)
            comm.Gather(block_energy_n, gather_energies, root=0)
            comm.Gather(block_observable_n, gather_observables, root=0)
            if options["ad_mode"] == "reverse":
                comm.Gather(block_rdm1_n, gather_rdm1s, root=0)
            elif options["ad_mode"] == "2rdm":
                comm.Gather(block_rdm2_n, gather_rdm2s, root=0)
        else:
            comm.Gather(block_weight_n, None, root=0)
            comm.Gather(block_energy_n, None, root=0)
            comm.Gather(block_observable_n, None, root=0)
            if options["ad_mode"] == "reverse":
                comm.Gather(block_rdm1_n, None, root=0)
            elif options["ad_mode"] == "2rdm":
                comm.Gather(block_rdm2_n, None, root=0)

        if rank == 0:
            block_slice = slice(n * size, (n + 1) * size)

            global_block_weights[block_slice] = gather_weights
            global_block_energies[block_slice] = gather_energies
            global_block_observables[block_slice] = gather_observables

            if options["ad_mode"] == "reverse":
                global_block_rdm1s[block_slice] = gather_rdm1s
            elif options["ad_mode"] == "2rdm":
                global_block_rdm2s[block_slice] = gather_rdm2s

            block_energy_n = (
                np.sum(gather_weights * gather_energies)
                / np.sum(gather_weights)
            )
        else:
            block_energy_n = 0.0

        block_energy_n = comm.bcast(block_energy_n, root=0)

        # do RO, global SR, update hat(E0) as usual
        prop_data = propagator.orthonormalize_walkers(prop_data)

        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data, f)

        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        # blocking analysis and print every 10% of superblocks
        # NOTE: hence best to have n_blocks be multiples of 10
        # dlai: I changed this to (n+1) to have correct behavior:
        # should print at the END of every 10% of blocks
        # this way it always prints n=n_blocks-1, the final block
        # and we see the stats at the final iter, properly
        if (n+1) % (max(propagator.n_blocks // 10, 1)) == 0:
            comm.Barrier()
            if rank == 0:
                e_afqmc, energy_error = stat_utils.blocking_analysis(
                    global_block_weights[: (n + 1) * size],
                    global_block_energies[: (n + 1) * size],
                    neql=0,
                )
                obs_afqmc, _ = stat_utils.blocking_analysis(
                    global_block_weights[: (n + 1) * size],
                    global_block_observables[: (n + 1) * size],
                    neql=0,
                )
                has_ad_mode = options["ad_mode"] is not None
                elapsed_time = time.time() - init
                sample_data = np.stack(
                    (
                        global_block_weights[: (n + 1) * size],
                        global_block_energies[: (n + 1) * size],
                        global_block_observables[: (n + 1) * size],
                    )
                ).T
                np.savetxt("samples_raw.dat", sample_data)
                # print progress
                if energy_error is not None:
                    if has_ad_mode:
                        print(
                            f"{n + 1:5d}",
                            f"{e_afqmc:>16.9e}",
                            f"{energy_error:>16.9e}",
                            f"{obs_afqmc:>16.9e}",
                            f"{elapsed_time:>7.1f}",
                            flush=True,
                            sep=" " * 5,
                        )
                    else:
                        print(
                            f"{n + 1:5d}",
                            f"{e_afqmc:>16.9e}",
                            f"{energy_error:>16.9e}",
                            f"{elapsed_time:>7.1f}",
                            flush=True,
                            sep=" " * 5,
                        )
                else:
                    if has_ad_mode:
                        print(
                            f"{n + 1:5d}",
                            f"{e_afqmc:>16.9e}",
                            f"{'-':>16s}",
                            f"{obs_afqmc:>16.9e}",
                            f"{elapsed_time:>7.1f}",
                            flush=True,
                            sep=" " * 5,
                        )
                    else:
                        print(
                            f"{n + 1:5d}",
                            f"{e_afqmc:>16.9e}",
                            f"{'-':>16s}",
                            f"{elapsed_time:>7.1f}",
                            flush=True,
                            sep=" " * 5,
                        )
            comm.Barrier()

    global_large_deviations = np.array(0)
    comm.Reduce(
        [local_large_deviations, MPI.INT],
        [global_large_deviations, MPI.INT],
        op=MPI.SUM,
        root=0,
    )
    comm.Barrier()
    if rank == 0:
        print(f"#\n# Number of large deviations: {global_large_deviations}", flush=True)

    comm.Barrier()
    e_afqmc, e_err_afqmc = None, None
    if rank == 0:
        np.savetxt(
            "samples_raw.dat",
            np.stack(
                (global_block_weights, global_block_energies, global_block_observables)
            ).T,
        )
        if options["ad_mode"] is not None and options["ad_mode"] != "2rdm":
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                2,
            )
        else:
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                1,
            )

        print(
            f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
        np.savetxt("samples.dat", samples_clean)
        global_block_weights = samples_clean[:, 0]
        global_block_energies = samples_clean[:, 1]
        global_block_observables = samples_clean[:, 2]
        if options["ad_mode"] == "reverse":
            global_block_rdm1s = global_block_rdm1s[idx]
        elif options["ad_mode"] == "2rdm":
            global_block_rdm2s = global_block_rdm2s[idx]

        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            global_block_weights, global_block_energies, neql=0, printQ=True
        )
        if e_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
            sig_err = np.around(
                np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_e = np.around(e_afqmc, sig_dec)
            print(f"AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
        elif e_afqmc is not None:
            print(f"AFQMC energy: {e_afqmc}\n", flush=True)
            e_err_afqmc = 0.0

        if options["ad_mode"] is not None:
            obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                global_block_weights, global_block_observables, neql=0, printQ=False
            )
            if err_afqmc is not None:
                sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
                sig_err = np.around(
                    np.round(err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
                )
                sig_obs = np.around(obs_afqmc, sig_dec)
                print(
                    f"AFQMC observable: {sig_obs:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n"
                )
            elif obs_afqmc is not None:
                print(f"AFQMC observable: {obs_afqmc}\n", flush=True)
            if options["ad_mode"] == "reverse":
                # avg_rdm1 = np.einsum('i,i...->...', global_block_weights, global_block_rdm1s) / np.sum(global_block_weights)
                norms_rdm1 = np.array(list(map(np.linalg.norm, global_block_rdm1s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm1)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm1s = global_block_rdm1s[idx]
                avg_rdm1 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm1s
                ) / np.sum(global_block_weights)
                errors_rdm1 = np.array(
                    list(map(np.linalg.norm, global_block_rdm1s - avg_rdm1))
                ) / np.linalg.norm(avg_rdm1)
                print(f"# RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm1, neql=0, printQ=False
                )
                np.savez("rdm1_afqmc.npz", rdm1=avg_rdm1)
            elif options["ad_mode"] == "2rdm":
                norms_rdm2 = np.array(list(map(np.linalg.norm, global_block_rdm2s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm2)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm2s = global_block_rdm2s[idx]
                avg_rdm2 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm2s
                ) / np.sum(global_block_weights)
                errors_rdm2 = np.array(
                    list(map(np.linalg.norm, global_block_rdm2s - avg_rdm2))
                ) / np.linalg.norm(avg_rdm2)
                print(f"# 2RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm2, neql=0, printQ=True
                )
                np.savez(
                    "rdm2_afqmc.npz",
                    rdm2=2 * avg_rdm2.reshape(ham.norb, ham.norb, ham.norb, ham.norb),
                )

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()
    return e_afqmc, e_err_afqmc

def fp_afqmc(
    ham_data, ham, propagator, trial, wave_data, observable, options, init_walkers=None
):
    init = time.time()
    seed = options["seed"]

    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)
    prop_data = propagator.init_prop_data(trial, wave_data, ham, ham_data, init_walkers)
    prop_data["key"] = jax.random.PRNGKey(seed + rank)

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("#\n# Sampling sweeps:")
        print("#  Iter        Mean energy          Stochastic error       Walltime")
    comm.Barrier()

    global_block_weights = np.zeros(size * propagator.n_ene_blocks) + 0.0j
    global_block_energies = np.zeros(size * propagator.n_ene_blocks) + 0.0j

    total_energy = np.zeros(propagator.n_blocks) + 0.0j
    total_weight = np.zeros(propagator.n_blocks) + 0.0j
    for n in range(
        propagator.n_ene_blocks
    ):  # hacking this variable for number of trajectories
        (
            prop_data_tr,
            energy_samples,
            weights,
            prop_data["key"],
        ) = sampler.propagate_free(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )
        global_block_weights[n] = weights[0]
        global_block_energies[n] = energy_samples[0]
        total_weight += weights
        total_energy += weights * (energy_samples - total_energy) / total_weight
        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data_tr, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data_tr, f)

        if n % (max(propagator.n_ene_blocks // 10, 1)) == 0:
            comm.Barrier()
            if rank == 0:
                print(f"{n:5d}: {total_energy}")
        np.savetxt(
            "samples_raw.dat", np.stack((global_block_weights, global_block_energies)).T
        )
