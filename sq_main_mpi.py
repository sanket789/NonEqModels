import numpy as np
import methods as func
import time
from mpi4py import MPI
import argparse
import math
'''
	This is main script for driven Aubry Andre Harper Model.
	The time dependent drive is in diagonal term with square wave protocol.
	The code is written in MPI framework to average over multiple disrder configurations
	The number of nodes in MPI must be chosen such that NUM_CONF is integral multiple of number of nodes.

	Sample command:
		Run: mpiexec -n 2 python3  sq_main_mpi.py -L 4 -tf 0.2 -NUM_CONF 2 -w 4.0 -mu0_h 3.0 -mu0_l 1.5 -dt 0.125
		
'''

def main_routine(arg,c,start_time):
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Extract simulation parameters
	J = 1.0
	L = arg.L
	w = arg.w
	PERIOD = 2*np.pi/w
	mu0_h = arg.mu0_h
	mu0_l = arg.mu0_l
	NUM_CONF = arg.NUM_CONF
	tf = arg.tf
	dt = arg.dt
	nT = int(tf/dt)
	if L < 20:
		l0 = 0
		l1 = L//2
	else:
		l0 = L//2 - 5
		l1 = L//2 + 5
	sigma = 0.5*(1.+np.sqrt(5.))
	save_data = True

	#Obtain alpha values for this thread
	alpha_min = mpi_rank*(2*np.pi)/mpi_size
	alpha_max = (mpi_rank+1)*(2*np.pi)/mpi_size
	num_alpha = NUM_CONF//mpi_size
	m_alpha = np.linspace(alpha_min,alpha_max,num_alpha,endpoint=False)

	#initialize correlation matrix
	CC_0 = np.zeros((L,L),dtype=complex)
	for j in range(0,L,2):
		CC_0[j,j] = 1.0

	#Initialize the storage matrices
	T = [dt*n for n in range(nT)]
	fname = "WDIR/L_%d/Nov26/Data/SQ_CHEM_L_%d_J_%g_w_%g_mu0H_%g_mu0L_%g_s_%g_conf_%d_tf_%g_dt_%g_" \
				%(L,L,J,w,mu0_h,mu0_l,sigma,NUM_CONF,tf,dt)

	m_S = np.zeros((num_alpha,nT))
	m_eigCC = np.zeros((num_alpha,nT,l1-l0))
	m_energy = np.zeros((num_alpha,nT))
	m_nbar = np.zeros((num_alpha,nT))
	m_imbalance = np.zeros((num_alpha,nT))
	m_current = np.zeros((num_alpha,nT,L))
	m_nn_site = np.zeros((num_alpha,nT))
	m_subsystem = np.zeros((num_alpha,nT,l1-l0,l1-l0),dtype=complex)
	m_pulse = np.zeros((num_alpha,nT))
	pulse = 0
	#Loop over nT
	for k in range(num_alpha):
		
		alpha = m_alpha[k]
		
		'''
		High State
		'''
		HH_1 = func.ham_diagonal_sq_drive_PBC(L,J,mu0_h,sigma,alpha)
		v_eps_1 , DD_1 = np.linalg.eigh(HH_1)
		EE_1 = np.diag(np.exp(-1j*dt*v_eps_1))
		UU_1 = np.dot(np.conj(DD_1),np.dot(EE_1,DD_1.T))
		'''
			Low State
		'''
		HH_2 = func.ham_diagonal_sq_drive_PBC(L,J,mu0_l,sigma,alpha)
		v_eps_2 , DD_2 = np.linalg.eigh(HH_2)
		EE_2 = np.diag(np.exp(-1j*dt*v_eps_2))
		UU_2 = np.dot(np.conj(DD_2),np.dot(EE_2,DD_2.T))

		CC_t = CC_0.copy()

		for i in range(nT):
			if math.fmod(T[i],PERIOD) < 0.5*PERIOD:	#High state
				HH = HH_1.copy()
				UU = UU_1.copy()
				pulse = 1
			else:	#Low state
				HH = HH_2.copy()
				UU = UU_2.copy()
				pulse = 0
			#Calculate Hamiltonian 
			phi_hop = 0.0
			
			#Calculate entropy

			m_subsystem[k,i,:]=CC_t[l0:l1,l0:l1]
			entropy, spectrum = func.vonNeumann_entropy(CC_t[l0:l1,l0:l1])
			m_S[k,i] = entropy
			m_eigCC[k,i,:] = spectrum.copy()
			#Calculate energy
			m_energy[k,i] = np.sum(np.multiply(HH,CC_t)).real
			#Calculate occupation of subsystem
			diagonal = np.diag(CC_t).real
			m_nbar[k,i] = np.sum(diagonal[l0:l1])/(l1-l0)
			#Calculate imbalance in entire system
			n_e = np.sum(diagonal[range(0,L,2)])	#at position 0,2,4, ... ,L-1
			n_o = np.sum(diagonal[range(1,L,2)])	#at position 1,3,5 ...
			m_imbalance[k,i] = (n_e-n_o)/(n_e+n_o)
			#Calculate current
			m_current[k,i,:] = func.charge_current(J,phi_hop,CC_t)
			#Calculate onsite occupation	
			m_nn_site[k,i] = CC_t[L//2,L//2].real
			
			#Pulse
			m_pulse[k,i] = pulse

			#Time evolution of correlation matrix. Refer readme for formulae
			CC_next = np.dot(np.conj(UU.T),np.dot(CC_t,UU))
			CC_t = CC_next.copy()
			
		print('Done alpha = %g'%m_alpha[k])

	'''
		Declare recv_ variable to gather the data from all nodes. General shape of variales is 
	'''
	recv_S = None
	recv_eigCC = None
	recv_nbar = None
	recv_imbalance = None
	recv_current = None
	recv_energy = None
	recv_nn_site = None
	recv_alpha = None
	recv_subsystem = None
	recv_pulse = None
	if mpi_rank == 0:
		recv_S = np.empty([mpi_size,num_alpha,nT])
		recv_eigCC = np.empty([mpi_size,num_alpha,nT,l1-l0])
		recv_nbar = np.empty([mpi_size,num_alpha,nT])
		recv_imbalance = np.empty([mpi_size,num_alpha,nT])
		recv_current = np.empty([mpi_size,num_alpha,nT,L])
		recv_energy = np.empty([mpi_size,num_alpha,nT])
		recv_nn_site = np.empty([mpi_size,num_alpha,nT])
		recv_alpha = np.empty([mpi_size,num_alpha])
		recv_subsystem = np.empty([mpi_size,num_alpha,nT,l1-l0,l1-l0],dtype=complex)
		recv_pulse = np.empty([mpi_size,num_alpha,nT])
	c.Gather(m_S,recv_S,root=0)
	c.Gather(m_eigCC, recv_eigCC,root=0)
	c.Gather(m_nbar,recv_nbar,root=0)
	c.Gather(m_imbalance,recv_imbalance,root=0)
	c.Gather(m_current,recv_current,root=0)
	c.Gather(m_energy,recv_energy,root=0)
	c.Gather(m_nn_site,recv_nn_site,root=0)
	c.Gather(m_alpha,recv_alpha,root=0)
	c.Gather(m_subsystem,recv_subsystem,root=0)
	c.Gather(m_pulse,recv_pulse,root=0)
	if mpi_rank == 0:
		#save data
		if save_data == True:
			np.save(fname+"entropy.npy",recv_S)
			np.save(fname+"CCspectrum.npy",recv_eigCC)
			np.save(fname+"nbar.npy",recv_nbar)
			np.save(fname+"imbalance.npy",recv_imbalance)
			np.save(fname+"current.npy",recv_current)
			np.save(fname+"energy.npy",recv_energy)
			np.save(fname+"nn_site.npy",recv_nn_site)
			np.save(fname+"conf.npy",recv_alpha)
			np.save(fname+"subsystem.npy",recv_subsystem)
			np.save(fname+"pulse.npy",recv_pulse)
			print('Data files saved successfully.')
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
		parser.add_argument("-L", "--L", help="System Size",default=512,type=int, nargs='?')
		parser.add_argument("-w", "--w", help="Frequency of drive",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0_h", "--mu0_h", help="Strength of chemical potential for High",default=2.0,type=float, nargs='?')
		parser.add_argument("-mu0_l", "--mu0_l", help="Strength of chemical potential for Low",default=2.0,type=float, nargs='?')
		parser.add_argument("-NUM_CONF","--NUM_CONF",help = "number of disorder config",default=100,type=int,nargs='?')
		parser.add_argument("-tf","--tf",help="total simulation time",default=1000,type=float,nargs='?')
		parser.add_argument("-dt","--dt",help="Simulation timestep",default=0.1,type=float,nargs='?')
		#-- add in the argument
		args=parser.parse_args()

	else:
		args = None

	# broadcast
	args = comm.bcast(args, root=0)

	# run main code
	main_routine(args, comm,start_time)
