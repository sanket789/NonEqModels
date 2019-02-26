import numpy as np
import methods as func
import time
from mpi4py import MPI
import argparse
# import matplotlib.pyplot as plt
from corr_init import ground_state
from time_evolution import evolve_correlation_matrix
'''
	This is main script for time evolution of Anderson insulator. The code is omptimized such that the diagonalization is done only 
	for single period and saved for subsequent times.  
	The code is written in MPI framework to average over multiple disorder configurations
	https://doi.org/10.1103/PhysRevB.98.214202	
'''


def main_routine(arg,c):
	start_time = time.time()
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Define simulation paramteres
	L = arg.L
	J = 1.0
	mu0 = arg.mu0
	A = arg.A
	w = arg.w
	num_steps = arg.num_steps
	num_period = arg.num_period
	num_conf = arg.num_conf
	nT = num_period*num_steps
	save_data = False
	dt = (2*np.pi/w)/num_steps
	T = [n*dt for n in range(nT)]
	'''
		#########################################  Main Code  ##############################################
	'''
	#Initial correlation matrix
	HH_start = func.ham_anderson_insl_PBC(L,J,A,np.zeros(L))
	eps_start, DD_start = np.linalg.eigh(HH_start)
	CC_0 = ground_state(L//2,HH_start)
	#Initialize the storage matrices
	fname="WDIR/Feb25/Data/ANDERSON_L_%d_mu0_%g_w_%g_steps_%d_tf_%g_ncnf_%g_"%(L,mu0,w,num_steps,num_period,num_conf*mpi_size)
	m_energy = np.zeros((num_conf,nT))		#energy expectation value
	m_nbar = np.zeros((num_conf,nT))		#Average occupation
	m_imb = np.zeros((num_conf,nT))			#Imbalance
	m_absb_energy = np.zeros((num_conf,nT))	#Absorbed energy as defined in ref
	m_diag = np.zeros((num_conf,nT,L))		#Diagonal of correlation matrix
	m_curr = np.zeros((num_conf,nT,L))		#Charge current
	m_excit = np.zeros((num_conf,L))		#Excitions as defined in ref
	'''
		#########################################  Loop over configurations and time  ##############################################
	'''
	for k in range(num_conf):
		mu_array = np.random.uniform(-mu0,mu0,L)
		UU_list = []
		HH_list = []
		for j in range(num_steps):
			phi = A*np.cos(w*T[j])
			HH = func.ham_anderson_insl_PBC(L,J,phi,mu_array)
			v_eps , DD = np.linalg.eigh(HH)
			EE = np.diag(np.exp(-1j*dt*v_eps))
			UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
			UU_list.append(UU)
			HH_list.append(HH)
		CC_t = CC_0.copy()

		for i in range(nT):
			#Calculate Hamiltonian
			HH_now = HH_list[i%num_steps]
			UU_now = UU_list[i%num_steps]
			m_nbar[k,i] = np.trace(CC_t).real		 
			m_imb[k,i] = func.imbalance(np.diag(CC_t).real)
			m_diag[k,i,:] = np.diag(CC_t).real
			m_energy[k,i] = np.sum(np.multiply(HH_now,CC_t)).real	
			m_absb_energy[k,i] = (m_energy[k,i] - m_energy[k,0])			
			m_curr[k,i,:] = func.charge_current(J,A*np.cos(w*T[i]),CC_t)
			CC_next = np.dot(np.conj(UU_now.T),np.dot(CC_t,UU_now))
			
			CC_t = CC_next.copy()

		m_absb_energy[k,:] = m_absb_energy[k,:]/(0.5*m_energy[k,-1]+0.5*m_energy[k,-2]-m_energy[k,0])
		#calculate excitations at time=tf
		for m in range(L):
			if eps_start[m] < 0:
				m_excit[k,m] = 1.0 - np.dot(DD_start.T,np.dot(CC_t,np.conj(DD_start)))[m,m].real
			else:
				m_excit[k,m] = np.dot(DD_start.T,np.dot(CC_t,np.conj(DD_start)))[m,m].real 
	'''
		############################	Gather the data to be saved 	##################################################
	'''
	recv_energy = None
	recv_nbar = None
	recv_imb = None
	recv_absb_energy = None
	recv_diag = None
	recv_curr = None
	recv_excit = None
	if mpi_rank	== 0:
		recv_energy = np.empty([mpi_size,num_conf,nT])
		recv_nbar = np.empty([mpi_size,num_conf,nT])
		recv_imb = np.empty([mpi_size,num_conf,nT])
		recv_absb_energy = np.empty([mpi_size,num_conf,nT])
		recv_diag = np.empty([mpi_size,num_conf,nT,L])
		recv_curr = np.empty([mpi_size,num_conf,nT,L])
		recv_excit = np.empty([mpi_size,num_conf,L])
	c.Gather(m_energy,recv_energy,root=0)
	c.Gather(m_nbar,recv_nbar,root=0)
	c.Gather(m_imb,recv_imb,root=0)
	c.Gather(m_absb_energy,recv_absb_energy,root=0)
	c.Gather(m_diag,recv_diag,root=0)
	c.Gather(m_curr,recv_curr,root=0)
	c.Gather(m_excit,recv_excit,root=0)
	if mpi_rank	== 0:
		recv_diag = np.mean(recv_diag,(0,1))
		recv_curr = np.mean(recv_curr,(0,1))
		if save_data == True:
			np.save(fname+"energy.npy",recv_energy)
			np.save(fname+"nbar.npy",recv_nbar)
			np.save(fname+"imb.npy",recv_imb)
			np.save(fname+"absb.npy",recv_absb_energy)
			np.save(fname+"diag.npy",recv_diag)
			np.save(fname+"curr.npy",recv_curr)
			np.save(fname+"excit.npy",recv_excit)
			print("Files successfully saved at : ",fname)
	end_time = time.time()
	print('Time taken by rank %d : %g seconds'%(mpi_rank,end_time - start_time))
'''
	#############################	Argument parsing 	#####################################
'''

if __name__ == '__main__':	
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	if rank==0:

		## parsing at rank =0
		parser = argparse.ArgumentParser(
				  description='Time evolution of fermion chain',
				  prog='main', usage='%(prog)s [options]')
		parser.add_argument("-L", "--L", help="System Size",default=100,type=int, nargs='?')
		parser.add_argument("-A","--A",help="Drive strength",default=.1,type=float,nargs='?')
		parser.add_argument("-num_steps","--num_steps",help="Discretization steps per drive period",default=10,type=int,nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential (disprder)",default=2.0,type=float, nargs='?')
		parser.add_argument("-num_conf","--num_conf",help = "number of disorder config per MPI_rank",default=100,type=int,nargs='?')
		parser.add_argument("-w","--w",help="Frequency",default=1.,type=float,nargs='?')
		parser.add_argument("-num_period","--num_period",help="Number of drive periods",default=10,type=int,nargs='?')

		#-- add in the argument
		args=parser.parse_args()
		print(args)

	else:
		args = None

	# broadcast
	bcasted_args = comm.bcast(args, root=0)

	# run main code
	main_routine(bcasted_args, comm)
