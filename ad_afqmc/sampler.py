import os

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import checkpoint, jit, lax, random

# import stat_utils
from ad_afqmc import linalg_utils

print = partial(print, flush=True)


@partial(jit, static_argnames=("propagator", "trial"))
def _step_scan(prop_data, fields, ham_data, propagator, trial, wave_data):
    """Wrapper for one step of ph-AFQMC. Called only in _block_scan()
    Probably unecessary... why not use propagator.propagate directly?"""
    prop_data = propagator.propagate(trial, ham_data, prop_data, fields, wave_data)
    return prop_data, fields


@partial(jit, static_argnames=("propagator", "trial"))
def _step_scan_free(prop_data, fields, ham_data, propagator, trial, wave_data):
    """Wrapper for one step of fp-AFQMC. Called only in _block_scan_free()
    WARN: propagate_free only supported by propagator_uhf class"""
    prop_data = propagator.propagate_free(trial, ham_data, prop_data, fields, wave_data)
    return prop_data, fields


N_STEPS_PER_CKPT = 5
# value 1 = original, unchanged

@partial(jit, static_argnames=("propagator", "trial"))
def _ckpt_step_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    """Propagate exactly N_STEPS_PER_CKPT steps. To be checkpointed.
    WARN: This impl changes the random number generation!
    Because the random fields are generated are now generated per CS block.
    This is necessary to prevent random fields from being saved as residuals,
    which will take up a huge amount of memory, larger than any residual.
    """
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            N_STEPS_PER_CKPT,
            propagator.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )
    prop_data, _ = lax.scan(
        partial(_step_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        prop_data,
        fields,
    )
    return prop_data, None


@partial(jit, static_argnames=("propagator", "trial"))
def _block_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    """
    Propagate n_prop_steps (50) steps of ph-AFQMC as a "block". (E-block)
    - WARN: bad naming, there are actually 3 layers of "blocks", this is the smallest one
    - random fields are generated for the entire block.
    - efficient impl using scan
    
    Performs actions done only once per E-block:
    - tally number of killed walkers
    - reorthonormalization
    - energy evaluation
        - with outlier rejection? via pop_control_ene_shift
        - DESPITE THE NAME, block energy and weights ARE SIMPLY THOSE OF THE LAST **STEP** IN THE BLOCK!!!
    - EMA update of pop_control_ene_shift this time with block energy
        - recall in each step, it is updated with inverse timestep * log average of new weights (?)
    
    Called:
    - by _sr_block_scan()       this is the "sr-block", one level up
    - by each nosr variant of phaseless_wrapper         when "sr-block" isn't needed
    """
    # # original impl: use when no step-ckpt needed
    # prop_data["key"], subkey = random.split(prop_data["key"])
    # fields = random.normal(
    #     subkey,
    #     shape=(
    #         propagator.n_prop_steps,
    #         propagator.n_walkers,
    #         ham_data["chol"].shape[0],
    #     ),
    # )
    # prop_data, _ = lax.scan(
    #     partial(_step_scan,
    #         ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
    #     prop_data,
    #     fields
    # )
    # # end of original impl
    prop_data, _ = lax.scan(
        checkpoint(partial(_ckpt_step_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data)),
        prop_data,
        None,
        length=propagator.n_prop_steps // N_STEPS_PER_CKPT,
    )
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    energy_samples = jnp.real(
        trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
    ) # computes local energy of all walkers
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["pop_control_ene_shift"])
        > jnp.sqrt(2.0 / propagator.dt),
        prop_data["pop_control_ene_shift"],
        energy_samples,
    ) # clips outlier walkers, if outlier falls back to baseline "pop_control_energy" WARN: sus behavior
    block_weight = jnp.sum(prop_data["weights"])
    block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight # the classic AFQMC equation, E = sum wE_L / sum w
    prop_data["pop_control_ene_shift"] = (
        0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
    )
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnames=("propagator", "trial"))
def _block_scan_free(prop_data, _x, ham_data, propagator, trial, wave_data):
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            propagator.n_prop_steps,
            propagator.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )
    prop_data, _ = lax.scan(
        partial(_step_scan_free,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        prop_data,
        fields,
    )
    energy_samples = trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
    # energy_samples = jnp.where(jnp.abs(energy_samples - ham_data['ene0']) > jnp.sqrt(2./propagator.dt), ham_data['ene0'], energy_samples)
    block_energy = jnp.sum(energy_samples * prop_data["overlaps"]) / jnp.sum(
        prop_data["overlaps"]
    )
    block_weight = jnp.sum(prop_data["overlaps"])
    return prop_data, (prop_data, block_energy, block_weight)


N_ENE_PER_CKPT = 1
# value 1 = original, unchanged

@partial(jit, static_argnames=("propagator", "trial"))
def _ckpt_block_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    """Propagate exactly nE/CKPT E-blocks. To be checkpointed."""
    prop_data, (block_energy, block_weight) = lax.scan(
        partial(
            _block_scan,
            ham_data=ham_data,
            propagator=propagator,
            trial=trial,
            wave_data=wave_data,
        ),
        prop_data,
        None,
        length=N_ENE_PER_CKPT,
    )

    return prop_data, (block_energy, block_weight)


# custom checkpointed version
@partial(jit, static_argnames=("propagator", "trial"))
def _sr_block_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    """
    Propagate n_ene_blocks E-blocks (each with n_prop_steps steps) as a "sr-block".
    - this is the second-tier block

    Additional actions done only once per sr-block:
    - local SR
        - NOTE: global SR is only done each "superblock", the next tier

    Called by: (NOTE: the following two impl are IDENTICAL)
    - _ad_block                         used in each SR-enabled variante of ad/nucgrad ph-AFQMC
    - propagate_phaseless_wrapper       vanilla ph-AFQMC
    """
    prop_data, (block_energy, block_weight) = lax.scan(
        partial(_ckpt_block_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        # checkpoint(partial(_ckpt_block_scan,
        #     ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data)),
        prop_data,
        None,
        length=propagator.n_ene_blocks // N_ENE_PER_CKPT,
    )

    # flatten the extra ckpt dimension
    block_energy = block_energy.reshape((-1, *block_energy.shape[2:]))
    block_weight = block_weight.reshape((-1, *block_weight.shape[2:]))

    prop_data = propagator.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnames=("propagator", "trial"))
def _ad_block(prop_data, ham_data, propagator, trial, wave_data):
    """
    Propagate n_sr_blocks SR-blocks (each with n_prop_steps * n_ene_blocks steps) as a "AD-block".
    - this is the superblock, but two more wrappers go around it
        - 1. the propagate_phaseless_[ad/nucgrad]_*
            - additionally does 
    
    Additional actions done only once per superblock:
    - 1. directly within _ad_block
        - reset killed walkers tally to 0; after superblock, computes walker death rate per ene_block (NOT per step)
            - this value seems to be unused afterwards
        - resets pop_control_ene_shift to energy estimate
            - but energy estimate gets updated each superblock via EMA with superblock energy so it's fine
    - 2. within the propagate wrapper
        - modifies H1 matrix to be the response-formalism (this modifies the Hamiltonian!)
            - modifies H2 matrix for 2rdm mode (and does chol on the spot)
            - modifies both H1 and H2 for nucgrad mode
        - seems to recreate the exp_1b operator among other things
            trial.optimize_orbs
            ham.rot_orbs
            ham.rot_ham
            ham.prop_ham
        - returns a ratio of sums, where no axis was supplied
            - If None, the sum is computed along all the axes.
            - so this returns a scalar... that is the weighted average over all E-block energies
                - the weights are the total walker weights at each E-block energy
                    - so if a block has more walker weight, its energy gets weighted more here
                    - is this reasonable?
                - recall that energy is only evaluated each E-block
                - then (n_sr_blocks, n_ene_blocks) of these energies are tallied by _ad_block
                - WARN: is this well defined? had a conversation with Yann and Yichi 8/14/2026
                - so what happens is that this equiv to summing up all over the union of walkers
                    from all E-block saved steps (say, step 100, 200, 300, 400, ..., 1000)
                    and then computing E = sum(wE_L) / sum(w) but this time summing over the UNION
                    - the math is E = sum(WE_AF) / sum(W) summing over all E-blocks
                     = sum(W sum(wE_L)/W) / sum(W)
                     = sum(sum(wE_L)) / sum(sum(w)) outer sum over E-blocks, inner sum over walkers
                    - my issue with this is that is treating walkers from different timesteps as the same
                    - OK this issue is resolved now... rationale is in "blocking analysis"
    - 3. within the main cycle
    - 
    """
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        partial(_sr_block_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        # checkpoint(partial(_sr_block_scan,
        #     ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data)),
        prop_data, None,
        length=propagator.n_sr_blocks,
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnames=("propagator", "trial"))
def _ad_block_nosr(prop_data, ham_data, propagator, trial, wave_data):
    """
    dlai: to reduce clutter.
    
    Diff from _ad_block:
    - use of _block_scan directly without the _sr_block_scan layer
    - WARN: ignores n_sr_blocks! Equivalent to setting it to 1.
    """
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        checkpoint(partial(_block_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data)),
        prop_data, None,
        length=propagator.n_ene_blocks,
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_ad(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    wave_data = trial.optimize_orbs(ham_data, wave_data)
    ham_data = ham.rot_orbs(ham_data, wave_data)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


# for 2-rdm
@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_ad_rdm2(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    # modify ham_data
    observable_op = (
        observable_op
        + jnp.transpose(observable_op, (2, 3, 0, 1))
        + jnp.transpose(observable_op, (1, 0, 3, 2))
        + jnp.transpose(observable_op, (3, 2, 1, 0))
    ) / 4.0
    norb = ham.norb
    ham_data["chol"] = linalg_utils.modified_cholesky(
        observable_op.reshape(norb**2, norb**2), norb, ham.nchol
    )
    wave_data = trial.optimize_orbs(ham_data, wave_data)
    ham_data = ham.rot_orbs(ham_data, wave_data)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_ad_nosr(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    wave_data = trial.optimize_orbs(ham_data, wave_data)
    ham_data = ham.rot_orbs(ham_data, wave_data)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block_nosr(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_ad_norot(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_ad_nosr_norot(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block_nosr(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless(ham, ham_data, propagator, prop_data, trial, wave_data):
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        partial(_sr_block_scan,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        prop_data,
        None,
        length=propagator.n_sr_blocks,
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_free(ham, ham_data, propagator, prop_data, trial, wave_data):
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data, (prop_data_tr, block_energy, block_weight) = lax.scan(
        partial(_block_scan_free,
            ham_data=ham_data, propagator=propagator, trial=trial, wave_data=wave_data),
        prop_data,
        None,
        length=propagator.n_blocks,
    )
    return prop_data_tr, block_energy, block_weight, prop_data["key"]


# section: nucgrad

@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_nucgrad_norot(
    ham, ham_data, rdm1op, rdm2op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = rdm1op
    rdm2op = (rdm2op + jnp.transpose(rdm2op, (0, 2, 1))) / 2
    ham_data["chol"] = rdm2op.reshape(-1, ham.norb * ham.norb)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_nucgrad_norot_nosr(
    ham, ham_data, rdm1op, rdm2op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = rdm1op
    rdm2op = (rdm2op + jnp.transpose(rdm2op, (0, 2, 1))) / 2
    ham_data["chol"] = rdm2op.reshape(-1, ham.norb * ham.norb)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block_nosr(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_nucgrad(
    ham, ham_data, rdm1op, rdm2op, propagator, prop_data, trial, wave_data
):
    # override hcore and chol integrals used in propagation
    # this doesn't affect the answer, but makes it an explicit function of the integrals
    ham_data["h1"] = rdm1op
    rdm2op = (rdm2op + jnp.transpose(rdm2op, (0, 2, 1))) / 2
    ham_data["chol"] = rdm2op.reshape(-1, ham.norb * ham.norb)

    # override the trial state with a version of itself, explicitly depending on the integrals
    # this is the most subtle and interesting part!
    # BEFORE: wave_data is the LAO-basis mo_coeff, read externally from HF
    # AFTER: wave_data is the LAO-basis mo_coeff, from SCF(ham_data[hcore, chol, dm0])
    mo_coeff = trial.optimize_orbs(ham_data, wave_data) # wave_data is only used as dm0
    wave_data = mo_coeff

    # precontract with trial, prepare 1b propagator
    # this used to be done before equilibration phase!
    # but now, we have the these intermediates fully differentiable wrt integrals
    # by having wave_data now a function of ham_data
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    # ph-AFQMC for n_sr_blocks * n_ene_blocks * n_prop_steps steps
    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnames=("ham", "propagator", "trial"))
def propagate_phaseless_nucgrad_rot_nosr(
    ham, ham_data, rdm1op, rdm2op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = rdm1op
    rdm2op = (rdm2op + jnp.transpose(rdm2op, (0, 2, 1))) / 2
    ham_data["chol"] = rdm2op.reshape(-1, ham.norb * ham.norb)

    mo_coeff = trial.optimize_orbs(ham_data, wave_data)
    wave_data = mo_coeff
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block_nosr(
        prop_data, ham_data, propagator, trial, wave_data
    )

    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data
    
