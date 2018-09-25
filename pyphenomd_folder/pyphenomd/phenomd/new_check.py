from astropy.io import ascii
import numpy as np 
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import pdb

data = np.genfromtxt('data_ready_june_snap_lim_1.txt', names=True)

z_coal = data['redshift']

Vc = 106.5**3

dz = 0.1
zs = np.arange(0.0, 5.0+dz, dz)
dz_dt = np.zeros_like(zs)
dVc_dz = np.zeros_like(zs)
dn_dzdVc = np.zeros_like(zs)
integrand = np.zeros_like(zs)
for i, z in enumerate(zs):
	inds = np.where((z_coal<z+dz)&(z_coal>=z))[0]
	dn_dzdVc[i] = -len(inds)/(dz*Vc)
	dz_dt[i] = dz/(cosmo.age(z+dz).value*1e9 - cosmo.age(z).value*1e9)
	dVc_dz[i] = (cosmo.comoving_volume(z+dz).value - cosmo.comoving_volume(z).value)/dz
	integrand[i] = dn_dzdVc[i]*dz_dt[i]*dVc_dz[i]/(1+z)
	print(i)

plt.plot(zs, integrand)
plt.show()
 
pdb.set_trace()