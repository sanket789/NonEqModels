import numpy as np
from scipy import special
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
import ED_func as f

L = 6	#system size
J = 1.
delta = 2.3
dJ = 0.1
T = 1.25
cyc = 100
nT = cyc
sigma = 0.5*(np.sqrt(5.)+1.)
mu_array = 0.5*np.asarray([np.cos(2*np.pi*sigma*nsite + 0.356) for nsite in range(L) ])
basis = f.construct_full_basis(L)
#Hamiltonian
Hchem = f.onsite_pot_op(basis,L,mu_array)		#onsite term
Hdel = f.pairing_op(basis,L,delta) 			#pairing term
HH1 = f.hop_op(basis,L,J+dJ) + Hchem + Hdel	#0 to T/2 Hamiltonian
HH2 = f.hop_op(basis,L,J-dJ) + Hchem + Hdel	#T/2 to T Hamiltonian
UU = f.getUnitary(HH1,HH2,0.5*T)
#State
# E_g,psi_g = sps.linalg.eigs(HH1,k=1,which='SR')
# psi_t = psi_g.T[0].copy()
# print(E_g, 'GS energy')
# print(psi_t)
psi_t = np.zeros(len(basis),dtype=complex)
psi_t[int(2*(2**L -1)/3.)] = 1.
print("Initial state : |" ,bin(basis[int(2*(2**L -1)/3.)])[2:].zfill(L),">")
# psi_t[15] = 1.0/np.sqrt(2)
# psi_t[12] = 1.0/np.sqrt(2)
# print("Initial state : ", bin(basis[15])[2:].zfill(L),bin(basis[12])[2:].zfill(L) )

# print(f.cdc_op(basis,L,1.,0,0).toarray())

#data storage
m_GG = np.zeros((nT,2*L,2*L),dtype=complex)
m_nb = np.zeros(nT)
m_energy = np.zeros(nT)
for i in range(nT):
	print(i*T)
	
	#calculate expectation values
	for j in range(L):
		for k in range(L):
			A = f.ccd_op(basis,L,1.,j,k)
			m_GG[i,j,k] = f.expect_val(A,psi_t)	
			A = f.cc_op(basis,L,1.,j,k)
			m_GG[i,j,L+k] = f.expect_val(A,psi_t)	
			A = f.cdcd_op(basis,L,1.,j,k)
			m_GG[i,L+j,k] = f.expect_val(A,psi_t)
			A = f.cdc_op(basis,L,1.,j,k)
			m_GG[i,L+j,L+k] = np.dot(psi_t.conj(),np.dot(A.toarray(),psi_t))#f.expect_val(A,psi_t)

	m_nb[i] = np.sum(np.diag(m_GG[i,L:2*L,L:2*L]).real)
	m_energy[i] = np.dot(psi_t.conj(),np.dot(HH1.toarray(),psi_t)).real #f.expect_val(HH1,psi_t).real
	#Evolution		
	psi_t = np.dot(UU,psi_t)
# print(m_GG[0,:,:].real)
# print(np.allclose(np.eye(L) - m_GG[:,0:L,0:L].conj(),m_GG[:,L:2*L,L:2*L]))
print('nb \n',m_nb)
# plt.plot(m_energy)
# plt.show()
np.save('ED/ED_mu.npy',mu_array)
np.save('ED/ED_GG.npy',m_GG)
print('energy \n',m_energy-np.sum(mu_array))
