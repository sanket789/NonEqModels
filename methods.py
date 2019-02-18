import numpy as np
import math
def phase(L,A,w,t):
	'''
		Input: 
			L : System Size
			A : Drive strength
			w : Drive frequency
			t : time
		Output:
			(A/L)*sin(wt)
	'''
	return (A/L)*np.sin(w*t)

def phase_constant_E(L,A,t):
	'''
		Input:	
			L : System Size
			A : Drive strength
			t : time
	'''
	return (A*t)/L


def ham_drive_pseudo_PBC(L,J,phi,mu0,sigma,alpha):
	'''
		Hamiltonian for non-interacting quasiperiodic driven fermion chain
		Input:
			L : System size
			J : Hopping parameter
			phi : Phase from Pierels' susbstitution
			mu0 : Chemical potential
			sigma : Irrational number
			alpha : Disorder parameter 
		Output:
			ham : L*L Hamiltonian matrix
	'''
	mu_n = [mu0*np.cos(2*np.pi*sigma*n + alpha) for n in range(L)]
	ham = np.diag(mu_n) - J*np.diag(np.exp(1j*phi)*np.ones(L-1),-1)  -J*np.diag(np.exp(-1j*phi)*np.ones(L-1),1)
	ham[0,-1] = -J*np.exp(1j*phi)
	ham[-1,0] = -J*np.exp(-1j*phi)

	return ham

def ham_diagonal_drive_PBC(L,J,phi,mu0,sigma,alpha):
	'''
		Hamiltonian for non-interacting quasiperiodic driven fermion chain. The time-dependent part is put in chemical potential term
		Input:
			L : System size
			J : Hopping parameter
			phi : Phase from Pierels' susbstitution
			mu0 : Chemical potential
			sigma : Irrational number
			alpha : Disorder parameter 
		Output:
			ham : L*L Hamiltonian matrix
	'''
	mu_n = [mu0*np.cos(2*np.pi*sigma*n + alpha) for n in range(L)]
	ham = phi*np.diag(mu_n) - J*np.diag(np.ones(L-1),-1)  -J*np.diag(np.ones(L-1),1)
	ham[0,-1] = -J
	ham[-1,0] = -J
	return ham

def ham_diagonal_sq_drive_PBC(L,J,mu0,sigma,alpha):
	'''
		Hamiltonian for non-interacting quasiperiodic driven fermion chain. The time-dependent part is put in chemical potential term
		The drive has square protocol
		mu(t) = mu0_h for 0 <= t < T/2
		mu(t) = mu0_l for T/2 <= t < T

		Input: 
			L : System size
			J : Hopping parameter
			mu0 : Chemical potential (depending on time)
			sigma : Irrational number
			alpha : Disorder parameter 
	'''
	mu_n = [mu0*np.cos(2*np.pi*sigma*n + alpha) for n in range(L)]
	ham = np.diag(mu_n) - J*np.diag(np.ones(L-1),-1)  -J*np.diag(np.ones(L-1),1)
	ham[0,-1] = -J
	ham[-1,0] = -J
	return ham	

def ham_anderson_insl_PBC(L,J,phi,mu_array):
	'''
		Hamiltonian for Anderson Insulator
		Input:
			L : System size
			J : Hopping parameter
			phi : Phase from Pierels' susbstitution
			mu0 : Chemical potential
		 
		Output:
			ham : L*L Hamiltonian matrix
	'''
	
	ham = np.diag(mu_array) - J*np.diag(np.exp(1j*phi)*np.ones(L-1),-1)  -J*np.diag(np.exp(-1j*phi)*np.ones(L-1),1)
	ham[0,-1] = -J*np.exp(1j*phi)
	ham[-1,0] = -J*np.exp(-1j*phi)

	return ham

def ham_DC_EField(L,J,phi):
	mu_n = np.arange(0,L)-L//2+0.5
	ham = -J*np.diag(np.ones(L-1),-1) - J*np.diag(np.ones(L-1),1) + phi*np.diag(mu_n)
	return ham 


def xlogx(x):
	try:
		S = x*math.log(x) + (1.-x)*math.log(1.-x)
	except ValueError as ve:
		S = 0.
	return S

def is_hermitian(AA):
	if np.allclose(np.conj(AA.T),AA):
		return True
	else:
		return False

def vonNeumann_entropy(AA):
	'''
		Input:
			AA : Correlation matrix
		Output:
			entropy : [real] vonNeumann entropy corresponding to AA
			evals : Eigen spectrum of AA
	'''
	evals = np.linalg.eigvalsh(AA)
	S = 0.
	for i in range(len(evals)):
		S = S - xlogx(evals[i]) 
	return S.real , evals

def charge_current(J,phi,AA):
	'''
		Input:
			J : Hopping parameter
			phi : time dependent phase
			AA : Correlation matrix
		Output:
			curr : Real valued 1D array of length shape(AA) [number of sites]
	'''
	L = np.shape(AA)[0]
	curr = np.zeros(L)
	curr[0] = (1j*J*(np.exp(1j*phi)*AA[0,L-1] - np.exp(-1j*phi)*AA[L-1,0])).real
	for i in range(1,L):
		curr[i] = (1j*J*(np.exp(1j*phi)*AA[i,i-1] - np.exp(-1j*phi)*AA[i-1,i])).real
	return curr


def imbalance(diag):
	'''
		Calculate (n_even-n_odd)/(n_even+n_odd)
		Input: 
			diag 	: 	Diagonal of Correlation matrix
		
	'''
	L = np.size(diag)
	n_even = np.sum(diag[range(0,L,2)])
	n_odd = np.sum(diag[range(1,L,2)])
	return (n_even-n_odd)/(n_even+n_odd)
