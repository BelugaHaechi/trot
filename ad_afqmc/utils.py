#import QMCUtils
import os
from pyscf import df,lib,scf,gto
import numpy as np
from ad_afqmc import pyscf_interface
from scipy.linalg import fractional_matrix_power

"""
Somehow all of these functions are unused :(

OK turns out this is bc a newer version of these exist in grad_utils.py
"""



def _ao2mo(chol,C): #Convert the 2e integrals from AO to MO basis
    chol2 = np.zeros((chol.shape[0], C.shape[0], C.shape[0]))
    for i in range(chol.shape[0]):
        chol2[i] = np.dot(C.T,np.dot(chol[i], C))
    return chol2

def write_df_integralsAFQMC(mf,h1=None,chol=None): #mf DF MF object
    """
    Given density-fitted MF object, extract integrals, convert from AO to MO, and write to disk.
    - writes to disk an H5 file called "FCIDUMP_chol" with keys:
    - hcore:        MO basis Hcore integrals
    - hcore_mod:    MO basis Hcore integrals minus "v0" remainder from H2 square decomposition (cost O(XK3))
    - chol:         MO basis CDERIs (efficiently transformed directly, O(XK3) cost instead of O(K5))
    - energy_core:  nuclear repulsion energy
    """
    mol = mf.mol
    if(chol is None):
        df0 = df.incore.cholesky_eri(mol,auxbasis=mf.auxbasis)#
        df0 = lib.unpack_tril(df0)
        chol1 = _ao2mo(df0,mf.mo_coeff)
    else:
        chol1 = _ao2mo(chol,mf.mo_coeff)
    #pyscf_interface.prep_afqmc(mf, chol_cut=chol_cut)#, integrals=integrals)
    
    if(h1 is not None): h1e = mf.mo_coeff.T @ h1 @ mf.mo_coeff
    else : h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    chol = chol1.copy()
    nbasis = h1e.shape[-1]
    nelec = mol.nelec
    enuc = mol.energy_nuc()
    chol = chol.reshape((-1,nbasis,nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij",chol,chol,optimize = "optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0],-1))
    #dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1()@ mf.get_ovlp() @ basis0
    q = np.linalg.qr(mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff)[0]
    #q = np.linalg.qr(dm0)[0][:,:sum(nelec)//2]
    #QMCUtils.writeMat(q, "rhf.txt")
    #np.savez("rhf.npz", mo_coeff=q)
    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,filename="FCIDUMP_chol")#,mo_coeffs=q)#[basis0,basis0])
    print("nchol:",chol.shape[0])
    print("nelec:",sum(nelec))

def write_df_lowdins_integralsAFQMC(mf,h1=None,chol=None):
    """
    Given density-fitted MF object, extract integrals,
    convert from AO to **Lowdin's orthogonalized AO basis**, and write to disk.
    - this is instead of the MO basis we're used to above!
    """
    mol = mf.mol
    basis0 = np.array(fractional_matrix_power(mf.get_ovlp(), -0.5),dtype="float64")
    X = basis0.copy()
    X_inv = np.linalg.inv(X)
    c_ao = X_inv @ mf.mo_coeff
    if(isinstance(mf,scf.uhf.UHF)):
        c_ao = [X_inv @ mf.mo_coeff[0], X_inv @ mf.mo_coeff[1]]
    elif(isinstance(mf,scf.rhf.RHF)):
        c_ao = X_inv @ mf.mo_coeff

    if(h1 is None): h1 = np.array(basis0.T @ mf.get_hcore() @ basis0, dtype="float64")
    else: h1 = h1 #np.array(basis0.T @ h1 @ basis0, dtype="float64")

    if(chol is None):
        df0 = df.incore.cholesky_eri(mol,auxbasis=mf.auxbasis)#,aosym='s1')
        df0 = lib.unpack_tril(df0)
        chol1 = _ao2mo(df0,basis0) #pyscf_interface.ao2mo_chol_copy(df0,mf.mo_coeff).reshape(df0.shape[0],mol.nao,mol.nao)
    else:
        chol1 = chol#_ao2mo(chol,basis0)
    
    print(chol1.shape)
    h1e = h1.copy()
    chol = chol1.copy()
    nbasis = h1e.shape[-1]
    nelec = mol.nelec
    enuc = mol.energy_nuc()
    chol = chol.reshape((-1,nbasis,nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij",chol,chol,optimize = "optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0],-1))
    q = c_ao #np.linalg.qr(dm0)[0]
    
    if(isinstance(mf,scf.uhf.UHF)):
        np.savez("uhf.npz", wave_data=q,mo_coeff=[basis0,basis0])
    elif(isinstance(mf,scf.rhf.RHF)):
        np.savez("rhf.npz", wave_data=q,mo_coeff=[basis0,basis0])
    
    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,filename="FCIDUMP_chol")#,mo_coeffs=[basis0,basis0])
    print("nchol:",chol.shape[0])
    print("nelec:",sum(nelec))


def weighted_average_matrices(matrices, weights):
    # Ensure weights is a numpy array
    weights = np.array(weights)
    
    # Calculate the weighted sum of the matrices
    weighted_sum = np.tensordot(weights, matrices, axes=(0, 0))
    
    # Calculate the sum of the weights
    sum_weights = np.sum(weights)
    
    # Calculate the weighted average
    weighted_average = weighted_sum / sum_weights
    
    return weighted_average

def find_nproc():
    import glob
    import re

    # Step 1: Get the list of files
    files = glob.glob('en_der_afqmc_*.npz')
    pattern = re.compile(r'en_der_afqmc_(\d+)\.npz')

    indices = []
    for file in files:
        match = pattern.search(file)
        if match:
            indices.append(int(match.group(1)))
    #print(max(indices)+1)
    return max(indices)+1

def get_rdmsDer_rhf(norb,nchol,filename="en_der_afqmc.npz",nproc = 4):
    """Unused..."""
    rdm1 = []
    rdm2 = []
    weight = []
    for i in range(nproc):
        weighti = np.load(f"en_der_afqmc_{i}.npz")["weight"]
        #import pdb;pdb.set_trace()
        rdm1i = np.load(f"en_der_afqmc_{i}.npz")["rdm1"].reshape(weighti.shape[0],norb,norb)
        rdm2i = np.load(f"en_der_afqmc_{i}.npz")["rdm2"].reshape(weighti.shape[0],nchol,norb,norb)
        for j in range(rdm1i.shape[0]):
            rdm1.append(rdm1i[j])
            rdm2.append(rdm2i[j])
            weight.append(weighti[j])
    #import pdb;pdb.set_trace()
    return np.array(rdm1),np.array(rdm2),np.array(weight)

from pyscf import ao2mo
def hf_custom(mf2,integrals, n_orbs, n_elec, dm_0=None,verbose=None):
    """
    Custom HF mf object that converges SCF with hardcoded integrals!
    - this effectively creates HF mo_coeff as a function of integrals, which is useful for AD
    
    Currently unused, but seems to be useful
    """
    mol = gto.Mole()
    mol.nelectron = sum(n_elec)
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: integrals["h0"]
    mol.build()
    mf = scf.RHF(mol) #df.density_fit(scf.RHF(mol),auxbasis=mf2.auxbasis)
    mf.energy_nuc = lambda *args: integrals["h0"]
    h1 = np.zeros((n_orbs, n_orbs))
    #print(integrals["h1"].shape)
    h1[np.tril_indices(n_orbs)] = integrals["h1"]
    h1 = h1 + h1.T - np.diag(h1.diagonal())
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n_orbs)
    eri0 = np.einsum("pij,pkl->ijkl",integrals["h2"],integrals["h2"])
    eri0 = ao2mo.restore(8,eri0,n_orbs)
    #print(eri0.shape)
    mf._eri = eri0 #ao2mo.restore(8, integrals["h2"], n_orbs)
    mf.max_cycle = 500
    if (verbose is not None):mf.verbose = verbose
    mf.kernel(dm0=dm_0)
    return mf

from ad_afqmc import stat_utils,grad_utils
def calc_observable(rdm1,rdm2,weights,obs1,obs2):

    val1 = np.einsum("wij,ij->w",rdm1,obs1)
    val2 = np.einsum("wijk,ijk->w",rdm2,obs2)
    val = val1 + val2
    data, mask = grad_utils.reject_outliers(val,m=10)
    if(np.sum(mask == False)):
        print(f"Outliers removed: {np.sum(mask == False)}")
    grad,err = stat_utils.blocking_analysis(weights[mask],data,neql=0,printQ=False,writeBlockedQ=False)
    return grad,err
