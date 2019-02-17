import numpy as np

'''
	The functions for initializing correlation matrix of real space correlators.
	C[i,j] = c^+_i c_j at t=0
	where i and j are spatial site indices.
'''

def left_spatial_occ(L):
	'''
	Initial state is such that left half is occupied. psi0 = |11110000>
		Input: L = number of sites
	'''
	AA = np.zeros((L,L),dtype=complex)
	for j in range(L//2):
		AA[j,j] = 1.0
	return AA

def ground_state(num,HH):
	'''
		Initial state is the ground state of the Hamiltonian.
			num	:	number of particles
			HH	: 	The Hamiltonian at time t=0
	'''
	eigval,eigvec = np.linalg.eigh(HH)
	AA = np.dot(np.conj(eigvec)[:,0:num],eigvec.T[0:num,:])
	return AA
