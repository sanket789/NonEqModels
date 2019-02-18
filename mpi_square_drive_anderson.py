import numpy as np
import methods as func
import time
from mpi4py import MPI
import argparse
# import matplotlib.pyplot as plt
from corr_init import ground_state
from time_evolution import evolve_correlation_matrix
'''
	This is main script for time evolution of Anderson insulator.  
	The code is written in MPI framework to average over multiple disorder configurations
	https://doi.org/10.1103/PhysRevB.98.214202	
'''
def ham(J,dJ,L,chem):
	HH = np.diag(chem) - (J+dJ)*np.diag(np.ones(L-1),1)- (J+dJ)*np.diag(np.ones(L-1),-1)
	HH[-1,0] = -(J+dJ)
	HH[0,-1] = -(J+dJ)
	return HH

def main_routine(arg,c):
	start_time = time.time()
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Define simulation paramteres
	L = arg.L
	J = 1.0
	mu0 = arg.mu0
	dJ = arg.delJ*J
	T = arg.T
	cyc = arg.cyc
	num_conf = arg.num_conf
	nT = 2*cyc
	save_data = True

	'''
		#########################################  Main Code  ##############################################
	'''
	#Initial correlation matrix
	HH_start = ham(J,dJ,L,np.zeros(L))
	CC_0 = ground_state(L//2,HH_start)
	#Initialize the storage matrices
	fname = "WDIR/Feb17/Data/SQ_ANDERSON_L_%d_dJ_%g_mu0_%g_T_%g_cyc_%d_ncnf_%g_"%(L,dJ,mu0,T,cyc,num_conf*mpi_size)
	m_energy = np.zeros((num_conf,nT))
	m_nbar = np.zeros((num_conf,nT))
	m_imb = np.zeros((num_conf,nT))
	m_absb_energy = np.zeros((num_conf,nT))
	'''
		#########################################  Loop over configurations and time  ##############################################
	'''
	for k in range(num_conf):
		mu_array = np.random.uniform(-mu0,mu0,L)
		#Define two Hamiltonians
		#Hamiltonian with (J+dJ)
		HH_h = ham(J,dJ,L,mu_array)
		v_eps_h , DD_h = np.linalg.eigh(HH_h)
		EE_h = np.diag(np.exp(-1j*0.5*T*v_eps_h))
		UU_h = np.dot(np.conj(DD_h),np.dot(EE_h,DD_h.T))
		#Hamiltonian with (J-dJ)
		HH_l = ham(J,-dJ,L,mu_array)
		v_eps_l , DD_l = np.linalg.eigh(HH_l)
		EE_l = np.diag(np.exp(-1j*0.5*T*v_eps_l))
		UU_l = np.dot(np.conj(DD_l),np.dot(EE_l,DD_l.T))
		#initialize correlation matrix to the ground state of HH_start		
		CC_t = CC_0.copy()

		for i in range(nT):
			#Calculate Hamiltonian
			m_nbar[k,i] = np.trace(CC_t).real		 
			m_imb[k,i] = func.imbalance(np.diag(CC_t).real)
			if i%2 == 0:
				m_energy[k,i] = np.sum(np.multiply(HH_h,CC_t)).real	
				m_absb_energy[k,i] = (m_energy[k,i] - m_energy[k,0])			
				CC_next = np.dot(np.conj(UU_h.T),np.dot(CC_t,UU_h))
			else:
				m_energy[k,i] = np.sum(np.multiply(HH_l,CC_t)).real
				m_absb_energy[k,i] = (m_energy[k,i] - m_energy[k,0])
				CC_next = np.dot(np.conj(UU_l.T),np.dot(CC_t,UU_l))
			CC_t = CC_next.copy()
		m_absb_energy[k,:] = m_absb_energy[k,:]/(0.5*m_energy[k,-1]+0.5*m_energy[k,-2]-m_energy[k,0])
	'''
		############################	Gather the data to be saved 	##################################################
	'''
	recv_energy = None
	recv_nbar = None
	recv_imb = None
	recv_absb_energy = None
	if mpi_rank	== 0:
		recv_energy = np.empty([mpi_size,num_conf,nT])
		recv_nbar = np.empty([mpi_size,num_conf,nT])
		recv_imb = np.empty([mpi_size,num_conf,nT])
		recv_absb_energy = np.empty([mpi_size,num_conf,nT])
	c.Gather(m_energy,recv_energy,root=0)
	c.Gather(m_nbar,recv_nbar,root=0)
	c.Gather(m_imb,recv_imb,root=0)
	c.Gather(m_absb_energy,recv_absb_energy,root=0)
	if mpi_rank	== 0:
		if save_data == True:
			np.save(fname+"energy.npy",recv_energy)
			np.save(fname+"nbar.npy",recv_nbar)
			np.save(fname+"imb.npy",recv_imb)
			np.save(fname+"absb.npy",recv_imb)
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
		parser.add_argument("-delJ","--delJ",help="Drive strength; hopping paramtere = (1+delJ)*J",default=.1,type=float,nargs='?')
		parser.add_argument("-T", "--T", help="Time period",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential",default=2.0,type=float, nargs='?')
		parser.add_argument("-num_conf","--num_conf",help = "number of disorder config per MPI_rank",default=100,type=int,nargs='?')
		parser.add_argument("-cyc","--cyc",help="Number of drive cycles",default=100,type=int,nargs='?')
		#-- add in the argument
		args=parser.parse_args()

	else:
		args = None

	# broadcast
	bcasted_args = comm.bcast(args, root=0)

	# run main code
	main_routine(bcasted_args, comm)
