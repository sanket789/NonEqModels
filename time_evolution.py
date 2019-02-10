import numpy as np

def evolve_correlation_matrix(dt,CC_init,v_eps,DD):
	'''
		Time evolution of correlation matrix from t=t0 to tf=dt. Please refer to the readme file for the detailed formula.
		Input:
			dt 		:	Time step for evolution
			CC_init	:	The correlation matrix at t = t0
			v_eps	:	eigenspectrum of the Hamiltonian under which unitary evolution is performed
			DD 		:	The column DD[:, i] is the normalized eigenvector corresponding to the eigenvalue v_eps[i]
	'''
	
	EE = np.diag(np.exp(-1j*dt*v_eps))
	UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
	CC_next = np.dot(np.conj(UU.T),np.dot(CC_init,UU))
	if not np.allclose(np.trace(CC_next.real) ,L//2):
		print('WARNING: The evolution of system is out of half filled subspace!')

	return CC_next