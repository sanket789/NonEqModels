import numpy as np
import methods as func
import time
from mpi4py import MPI
import argparse
import matplotlib.pyplot as plt
'''
	This is main script for time evolution of Anderson insulator.  
	The code is written in MPI framework to average over multiple disorder configurations
	https://doi.org/10.1103/PhysRevB.98.214202	
'''

def main_routine(arg,c,start_time):
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Extract simulation parameters
	J = 1.0
	L = arg.L
	w = arg.w
	mu0 = arg.mu0
	A = arg.A
	NUM_CONF = arg.NUM_CONF
	num_cyc = arg.num_cyc
	num_samp = arg.num_samp
	dt = (2*np.pi/w)/num_samp
	nT = int(num_cyc*num_samp)
	if L < 20:
		l0 = 0
		l1 = L//2-1
	else:
		l0 = L//2 - 5
		l1 = L//2 + 6
	
	save_data = False
	indx = np.zeros(L)
	for i in range (L//2):
		indx[2*i] = i
		indx[2*i+1] = -1-i
	
	
	
	#initialize correlation matrix to the ground state of clean Hamiltonian at t=0
	CC_0 = np.zeros((L,L),dtype=complex)
	HH0 = func.ham_anderson_insl_PBC(L,J,A,np.zeros(L))
	eps0,DD0 = np.linalg.eigh(HH0)
	CC_0 = np.dot(np.conj(DD0)[:,0:L//2],DD0.T[0:L//2,:])
	#for p in range(L):
	#	for q in range(L):
	#		CC_0[p,q] = (1./L)*sum([np.exp(-1j*2*np.pi*indx[n]*(p-q)/L) for n in range(L//2)])
			#if j==0 and k==0:
			#	CC_0[j,k] = L/4.
			#elif k==0:
			#	CC_0[j,k] = 0.5*np.exp(-1j*np.pi*j*0.5)*(np.exp(1j*np.pi*j)-1)/(np.exp(1j*np.pi*j*2/L)-1)
			#elif j == 0:
			#	CC_0[j,k] = 0.5*np.exp(1j*np.pi*k*0.5)*(np.exp(-1j*np.pi*k)-1)/(np.exp(-1j*np.pi*k*2/L)-1)
			#else:
			#	CC_0[j,k] =(1/L)*np.exp(1j*np.pi*(k-j)*0.5)*(np.exp(1j*np.pi*j)-1)*(np.exp(-1j*np.pi*k)-1)/ \
			#			((np.exp(1j*np.pi*j*2/L)-1)*(np.exp(-1j*np.pi*k*2/L)-1))
			
	#print(func.is_hermitian(CC_0))

	#Initialize the storage matrices
	T = [dt*n for n in range(nT)]
	fname = "WDIR/L_100/Jan23/AND_L_%d_A_%g_w_%g_mu0_%g_num_%d_cycles_%g_samples_%g_"%(L,A,w,mu0,NUM_CONF,num_cyc,num_samp)
	num_mu0 = NUM_CONF//mpi_size
	m_S = np.zeros((num_mu0,nT))
	m_eigCC = np.zeros((num_mu0,nT,l1-l0))
	m_energy = np.zeros((num_mu0,nT))
	m_nbar = np.zeros((num_mu0,nT))
	m_imbalance = np.zeros((num_mu0,nT))
	m_current = np.zeros((num_mu0,nT,L))
	m_nn_site = np.zeros((num_mu0,nT))
	m_subsystem = np.zeros((num_mu0,nT,l1-l0,l1-l0),dtype=complex)
	m_deln = np.zeros((num_mu0,L))
	m_Einf = np.zeros((num_mu0))
	#Loop over nT
	for k in range(num_mu0):
		CC_t = CC_0.copy()
		mu_array = np.random.uniform(-1.0*mu0,mu0,L)
		m_Einf[k] = 2**(L-1)*np.sum(mu_array)
		for i in range(nT):
			#print (100.*i/nT)
			#Calculate Hamiltonian			 
			phi_hop =A*np.cos(w*T[i])
			
			HH = func.ham_anderson_insl_PBC(L,J,phi_hop,mu_array)
			#Calculate entropy
			m_subsystem[k,i,:]=CC_t[l0:l1,l0:l1]
			entropy, spectrum = func.vonNeumann_entropy(CC_t[l0:l1,l0:l1])
			m_S[k,i] = entropy
			m_eigCC[k,i,:] = spectrum.copy()
			#Calculate energy
			m_energy[k,i] = np.sum(np.multiply(HH,CC_t)).real
			#Calculate occupation of subsystem
			diagonal = np.diag(CC_t).real
			m_nbar[k,i] = np.sum(diagonal)
			#Calculate imbalance in entire system
			n_e = np.sum(diagonal[range(0,L,2)])	#at position 0,2,4, ... ,L-1
			n_o = np.sum(diagonal[range(1,L,2)])	#at position 1,3,5 ...
			m_imbalance[k,i] = (n_e-n_o)/(n_e+n_o)
			#Calculate current
			m_current[k,i,:] = func.charge_current(J,phi_hop,CC_t)
			#Calculate onsite occupation	
			m_nn_site[k,i] = CC_t[L//2,L//2].real
			
			#Time evolution of correlation matrix. Refer readme for formulae
			v_eps , DD = np.linalg.eigh(HH)
			EE = np.diag(np.exp(-1j*dt*v_eps))
			UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
			CC_next = np.dot(np.conj(UU.T),np.dot(CC_t,UU))
			CC_t = CC_next.copy()
		for m in range(L):
			if eps0[m] < 0:
				m_deln[k,m] = 1.0 - np.dot(DD0.T,np.dot(CC_t,np.conj(DD0)))[m,m].real
			else:
				m_deln[k,m] = np.dot(DD0.T,np.dot(CC_t,np.conj(DD0)))[m,m].real 
		

	'''
		Declare recv_ variable to gather the data from all nodes. General shape of variales is 
	'''
	# recv_S = None
	# recv_eigCC = None
	recv_nbar = None
	# recv_imbalance = None
	# recv_current = None
	recv_energy = None
	# recv_nn_site = None
	# recv_subsystem = None
	recv_deln = None
	recv_Einf = None
	if mpi_rank == 0:
		# recv_S = np.empty([mpi_size,num_mu0,nT])
		# recv_eigCC = np.empty([mpi_size,num_mu0,nT,l1-l0])
		recv_nbar = np.empty([mpi_size,num_mu0,nT])
		# recv_imbalance = np.empty([mpi_size,num_mu0,nT])
		# recv_current = np.empty([mpi_size,num_mu0,nT,L])
		recv_energy = np.empty([mpi_size,num_mu0,nT])
		# recv_nn_site = np.empty([mpi_size,num_mu0,nT])
		recv_deln = np.empty([mpi_size,num_mu0,L])
		# recv_subsystem = np.empty([mpi_size,num_mu0,nT,l1-l0,l1-l0],dtype=complex)
		recv_Einf = np.empty([mpi_size,num_mu0])
	# c.Gather(m_S,recv_S,root=0)
	# c.Gather(m_eigCC, recv_eigCC,root=0)
	c.Gather(m_nbar,recv_nbar,root=0)
	# c.Gather(m_imbalance,recv_imbalance,root=0)
	# c.Gather(m_current,recv_current,root=0)
	c.Gather(m_energy,recv_energy,root=0)
	# c.Gather(m_nn_site,recv_nn_site,root=0)
	c.Gather(m_deln,recv_deln,root=0)
	# c.Gather(m_subsystem,recv_subsystem,root=0)
	c.Gather(m_Einf,recv_Einf,root=0)
	if mpi_rank == 0:
		print(np.allclose(m_nbar,0.5*L*np.ones((mpi_size,num_mu0,nT)))) 
		if save_data == True:
			# np.save(fname+"entropy.npy",recv_S)
			# np.save(fname+"CCspectrum.npy",recv_eigCC)
			np.save(fname+"nbar.npy",recv_nbar)
			# np.save(fname+"imbalance.npy",recv_imbalance)
			# np.save(fname+"current.npy",recv_current)
			np.save(fname+"energy.npy",recv_energy)
			# np.save(fname+"nn_site.npy",recv_nn_site)
			np.save(fname+"deln.npy",recv_deln)
			# np.save(fname+"subsystem.npy",recv_subsystem)
			np.save(fname+"Einf.npy",recv_Einf)
			print('Data files saved successfully.')
			print('Filename : ',fname)
		end_time = time.time()
		print("Total Simulation time = ",end_time-start_time," seconds")



if __name__ == '__main__':	
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	start_time = time.time()
	if rank==0:

		## parsing at rank =0
		parser = argparse.ArgumentParser(
				  description='Time evolution of fermion chain',
				  prog='main', usage='%(prog)s [options]')
		parser.add_argument("-L", "--L", help="System Size",default=100,type=int, nargs='?')
		parser.add_argument("-w", "--w", help="Frequency of drive",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential",default=2.0,type=float, nargs='?')
		parser.add_argument("-A", "--A", help="Strength of drive",default=1.0,type=float, nargs='?')
		parser.add_argument("-NUM_CONF","--NUM_CONF",help = "number of disorder config",default=100,type=int,nargs='?')
		parser.add_argument("-num_cyc","--num_cyc",help="total number of cycles",default=40,type=float,nargs='?')
		parser.add_argument("-num_samp","--num_samp",help="samples in one period",default=100,type=float,nargs='?')
		#-- add in the argument
		args=parser.parse_args()

	else:
		args = None

	# broadcast
	args = comm.bcast(args, root=0)

	# run main code
	main_routine(args, comm,start_time)
