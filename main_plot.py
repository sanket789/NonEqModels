import numpy as np
import matplotlib.pyplot as plt
import plot_func as myplt

def disorder_avg(raw_data):
	#reshape raw_data sets by merging first two axis
	init_shape = raw_data.shape
	if len(init_shape) > 2:
		temp = np.average(raw_data,0)
		avg_data = np.average(temp,0)
	else:
		avg_data = raw_data.copy()
	return avg_data


show = False
cl = ['blue', 'brown', 'green','magenta']
#import  18 directories
L = 512

J = 1.0
dt = 0.1
tf = 1000.

sigma = 0.5*(np.sqrt(5) + 1)	#irrational number for quasi-periodicity

N = int(tf/dt)
T = [dt*n for n in range(N)]

w0 = 0.0483322
w1 = 16.0

A_W = [[32.0 , w0],[32.0,w1],[512.0,w0],[512.0,w1]]
mu0 = [0.0,0.5,2.0,3.0]
NUM_CONF = [1.0,50.0,50.0,50.0]
name=[[],[],[],[]]
for j in range(4):
	for k in range(4):
		name[j].append("AVG_L_%d_J_%g_A_%g_w_%g_mu0_%g_s_%g_conf_%d_tf_%g_dt_%g_"
						%(L,J,A[k,0],w[k,1],mu0[j],sigma,NUM_CONF[j],tf,dt)) 

for i in range(4):
	print('mu0 = ' ,mu0[i])
	#load entropy and average it 
	S0 = disorder_avg(np.load(name[i,0]+"entropy.npy"))
	S1 = disorder_avg(np.load(name[i,1]+"entropy.npy"))
	S2 = disorder_avg(np.load(name[i,2]+"entropy.npy"))
	S3 = disorder_avg(np.load(name[i,3]+"entropy.npy"))

	myplt.multiplot(T,[S0,S1,S2,S3],x='Time in seconds',y='Entropy',ttl='Entropy of subsystem',lab=wlabel,
					fname=fn+'/entropy_N_512_mu0_%g.png'%(mu0[i]),save=save,show=show,color=cl)
	myplt.multiplot_log_log(T,[S0,S1,S2,S3],x='Time in seconds',y='Entropy',ttl='Entropy of subsystem',lab=wlabel,
		fname=fn+'/Logentropy_N_512_mu0_%g.png'%mu0[i],save=save, show=show,color=cl)
	#load average occupation and average it
	NB0 = disorder_avg(np.load(name[i,0]+"nbar.npy")) - 0.5
	NB1 = disorder_avg(np.load(name[i,1]+"nbar.npy")) - 0.5
	NB2 = disorder_avg(np.load(name[i,2]+"nbar.npy"))- 0.5
	NB3 = disorder_avg(np.load(name[i,3]+"nbar.npy"))- 0.5
	myplt.multiplot(T,[NB0,NB1,NB2,NB3],x='Time in seconds',y='Average ocuupation',
					ttl='Average occupation of subsystem',lab=wlabel,
					fname=fn+'/nbar_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	
	#load imbalance and average it 
	IM0 = disorder_avg(np.load(name[i,0]+"imbalance.npy"))
	IM1 = disorder_avg(np.load(name[i,1]+"imbalance.npy"))
	IM2 = disorder_avg(np.load(name[i,2]+"imbalance.npy"))
	IM3 = disorder_avg(np.load(name[i,3]+"imbalance.npy"))
	myplt.multiplot(T,[IM0,IM1,IM2,IM3]],x='Time in seconds',y='number',
					ttl='even odd imbalance in occupation',lab=wlabel,
					fname=fn+'/imbalance_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)

	#load energy expectation and average it
	E0 = disorder_avg(np.load(name[i,0]+"energy.npy"))
	E1 = disorder_avg(np.load(name[i,1]+"energy.npy"))
	E2 = disorder_avg(np.load(name[i,2]+"energy.npy"))
	E3 = disorder_avg(np.load(name[i,3]+"energy.npy"))
	myplt.multiplot(T,[(E0 - E0[0])/L,(E1 - E1[0])/L,(E2 - E2[0])/L,(E3 - E3[0])/L],
					x='Time in seconds',y='Energy',ttl='Energy expectation density(E(t) - E(t=0))',lab=wlabel,
					fname=fn+'/energy_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	
	#load site occupation and averge it
	NSITE0 = disorder_avg(np.load(name[i,0]+"nn_site.npy"))
	NSITE1 = disorder_avg(np.load(name[i,1]+"nn_site.npy"))
	NSITE2 = disorder_avg(np.load(name[i,2]+"nn_site.npy"))
	NSITE3 = disorder_avg(np.load(name[i,3]+"nn_site.npy"))
	
	myplt.multiplot(T,[NSITE0,NSITE1,NSITE2,NSITE3],
					x='Time in seconds',y='number',ttl='Occupation at site %d'%(L//2),lab=wlabel,
					fname=fn+'/nnsite_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	
	
	
	#load current and average it

	j0 = np.load(name[i,0]+"current.npy")
	j1 = np.load(name[i,1]+"current.npy")
	j2 = np.load(name[i,2]]+"current.npy")
	j3 = np.load(name[i,3]+"current.npy")
		

	#plot current at site L/2
	myplt.multiplot(T,[j0[:,L//2],j1[:,L//2],j2[:,L//2],j3[:,L//2]],x='Time in seconds',y='Charge current',
					ttl='Current at site %d'%(L//2),lab=wlabel,
					fname=fn+'/halfCurrent_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	#plot current at end of simulation
	myplt.multiplot(range(0,L),[j0[-1,:],j1[-1,:],j2[-1,:],j3[-1,:]],x='Site index',y='Charge current',
					ttl='Current at t =  %g'%tf,lab=wlabel,
					fname=fn+'/FinalCurrent_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	#plot total averaged current
	myplt.multiplot(T,[np.average(j0,1),np.average(j1,1),np.average(j2,1),np.average(j3,1)],
					x='Time in seconds',y='Total average charge current',
					ttl='Total average charge current',lab=wlabel,
					fname=fn+'/AvgCurrent_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
	#plot Fourier transform
	jw0 = f.fftshift(f.fft(np.average(j0,1)))
	jw1 = f.fftshift(f.fft(np.average(j1,1)))
	jw2 = f.fftshift(f.fft(np.average(j2,1)))
	jw3 = f.fftshift(f.fft(np.average(j3,1)))


	fftW = 2*np.pi*f.fftshift(f.fftfreq(N,dt))
	myplt.multiplot(fftW,[abs(jw0)**2,abs(jw1)**2,abs(jw2)**2,abs(jw3)**2],
		x='fourier frequency (w)',y='current fourier transform', ttl='Fourier mode',lab=wlabel,
		fname=fn+'/FFT_avg_current_N_512_mu0_%g.png'%mu0[i],save=save,show=show,color=cl)
