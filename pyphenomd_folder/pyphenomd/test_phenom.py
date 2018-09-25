from pyphenomd import PhenomDWaveforms
import pdb
import matplotlib.pyplot as plt
import scipy.constants as ct
import numpy as np
import time

def taper(f, amp, f1, f2):
	return 0.0*(f<=f1) + 1./(np.exp((f2-f1)/(f-f1) + (f2-f1)/(f-f2))+1.0)*((f>f1) & (f<f2))*amp + amp*(f>=f2)

def run_wave(num):
	m1 = m2 = np.full((num,), 1e6)
	chi1 = chi2 = np.full((num,), -.9)
	z = np.full((num,), 1.0)
	st = np.full((num,), 1.0)
	et = np.full((num,), 0.0)

	wave = PhenomDWaveforms(m1, m2, chi1, chi2, z, st, et, df=1e-8, dist_type='comoving_distance')#, num_points=5000001)
	wave.create_waveforms()
	pdb.set_trace()
	return wave

m1 = 1.0
m2 = 1.0

ts = []
nums = np.arange(1,10+1, 1)
for num in nums:
	end = 0
	for i in range(3):
		start = time.time()
		wave = run_wave(num)
		end += time.time() - start
	ts.append(end/3.)
	print(num)

wave = run_wave(1)

hp_f = wave.amplitude*np.exp(-1j*wave.phase)

pdb.set_trace()
#plt.loglog(wave.freqs, wave.amplitude)
plt.plot(nums, ts)
plt.xlabel('Number of Waveforms')
plt.ylabel('Avg. Time 3 iterations (sec)')
#plt.savefig('timing_waveforms_10K.pdf')
plt.show()
exit()

#M = (m1+m2)*1.989e30*ct.G/ct.c**2
f2 = 0.001

wave.tapered_amp = taper(wave.freqs, wave.amplitude, f2*0.8, f2)



hp_t = np.fft.ifft(hp_f)
#plt.plot(hp_t.imag)
plt.plot(wave.freqs, wave.phase)
plt.show()
pdb.set_trace()