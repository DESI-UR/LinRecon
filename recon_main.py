import numpy as np
from astropy.io import fits
from nbodykit.lab import *
from scipy import interpolate
import dask.array as da
from scipy.ndimage import gaussian_filter
from recon_functions import *

# Cosmology, files, and parameters needed for reconstruction
cosmo = cosmology.Cosmology(h=0.674).match(Omega0_m=0.315)
dat_file = ''
ran_file = ''
gridNumber = 200
bias = 1.2
growth_factor = (Omega0_m**0.6)/bias
# Convert to Cartesian and get grid indices for each galaxy
print('Converting to Cartesian coords')
ct_dat_pos,dat_den = survey2cartesian(dat_file,cosmo,0.01,'data')
ct_ran_pos,ran_den = survey2cartesian(ran_file,cosmo,0.01,'randoms')
boxmin = ct_ran_pos.min()
boxsize = ct_ran_pos.max()-boxmin
dat_index = get_grid_index(ct_dat_pos,boxmin,boxsize,gridNumber)
ran_index = get_grid_index(ct_ran_pos,boxmin,boxsize,gridNumber)
# Paint galaxies onto a density field, then an overdensity field
print('Creating overdensity field')
data_field = grid_assignment(dat_index,gridNumber,dat_den)
random_field = grid_assignment(ran_index,gridNumber,ran_den)
data_overdensity = get_overdensity(data_field)
random_overdensity = get_overdensity(random_field)
print('Each grid step is '+str(round(boxsize/gridNumber,2))+'Mpc/h')
# Smooth grid and solve for scalar field in fourier space
print('Smoothing')
fft = np.fft.fftn(gaussian_filter(data_overdensity, sigma=1))
wavenumber = getWavenumber(gridNumber,boxsize)
phi = np.fft.ifftn(-fft*(1/wavenumber**2)*(1/bias))
# Take the gradient of the scalar field for the displacement field
print('Computing displacement field')
qred = np.gradient(phi)
axis = np.arange(gridNumber)*(boxsize/gridNumber)
grid_lens = np.array(np.meshgrid(axis,axis,axis))
LOS_VECTORS = [grid_lens[0][1:,1:,1:]/np.sqrt(grid_lens[0][1:,1:,1:]**2+grid_lens[1][1:,1:,1:]**2+grid_lens[2][1:,1:,1:]**2),\
               grid_lens[1][1:,1:,1:]/np.sqrt(grid_lens[0][1:,1:,1:]**2+grid_lens[1][1:,1:,1:]**2+grid_lens[2][1:,1:,1:]**2),\
               grid_lens[2][1:,1:,1:]/np.sqrt(grid_lens[0][1:,1:,1:]**2+grid_lens[1][1:,1:,1:]**2+grid_lens[2][1:,1:,1:]**2)]
qreal = qred.copy()
qreal[0][1:,1:,1:] = qred[0][1:,1:,1:]*(1+growth_factor*np.abs(LOS_VECTORS[0]))
qreal[1][1:,1:,1:] = qred[1][1:,1:,1:]*(1+growth_factor*np.abs(LOS_VECTORS[1]))
qreal[2][1:,1:,1:] = qred[2][1:,1:,1:]*(1+growth_factor*np.abs(LOS_VECTORS[2]))
maxs = np.asarray([qreal[0].real.max(),qreal[1].real.max(),qreal[2].real.max()])
mins = np.asarray([qreal[0].real.min(),qreal[1].real.min(),qreal[2].real.min()])
avgs = [np.mean(abs(qreal[0][qreal[0] != 0].real)),np.mean(abs(qreal[1][qreal[1] != 0].real)),np.mean(abs(qreal[2][qreal[2] != 0].real))]
# Check that displacements are reasonable
print("Min/Max of displacement field:")
print("["+str(np.amin(mins))+","+str(np.amax(maxs))+"]")
print("Mean x,y,z displacement = "+str(avgs[0])+", "+str(avgs[1])+", "+str(avgs[2]))
# Shift galaxy positions and write them back into fits files for analysis
print('Shifting galaxies and writing files')
dataNew = shiftParticle(ct_dat_pos,qreal,gridNumber,boxsize,boxmin).T
randomNew = shiftParticle(ct_ran_pos,qreal,gridNumber,boxsize,boxmin).T
dataNew_dask = da.from_array(dataNew.T, chunks=(1000, 1000))
randomNew_dask = da.from_array(randomNew.T, chunks=(1000, 1000))
recon_data = transform.CartesianToSky(dataNew_dask, cosmo, velocity=None, observer=[0, 0, 0], zmax=1.)
recon_randoms = transform.CartesianToSky(randomNew_dask, cosmo, velocity=None, observer=[0, 0, 0], zmax=1.)
column_names = ['ra','dec','z','weight_fkp','weight_noz','weight_cp','weight_sdc','modelflux_0','modelflux_1','modelflux_2','modelflux_3','modelflux_4']
columns = []
new_col = fits.Column(name='ra', format='E', array=recon_data[0].compute())
columns.append(new_col)
new_col = fits.Column(name='dec', format='E', array=recon_data[1].compute())
columns.append(new_col)
new_col = fits.Column(name='z', format='E', array=recon_data[2].compute())
columns.append(new_col)
for column in column_names[3:12]:
    new_col = fits.Column(name=column, format='E', array=fits.open(dat_file)[1].data[column])
    columns.append(new_col)
new_hdus = fits.BinTableHDU.from_columns(columns)
new_hdus.writeto('/Users/zackbrown/Documents/Rochester_Physics/Demina_Research/data/recon_galaxy_DR12_fullweights_magnitudes.fits')
columns = []
idx = recon_randoms[0].compute()
new_col = fits.Column(name='ra', format='E', array=recon_randoms[0].compute())
columns.append(new_col)
new_col = fits.Column(name='dec', format='E', array=recon_randoms[1].compute())
columns.append(new_col)
new_col = fits.Column(name='z', format='E', array=recon_randoms[2].compute())
columns.append(new_col)
new_col = fits.Column(name='weight', format='E', array=fits.open(ran_file)[1].data['weight'])
columns.append(new_col)
new_hdus = fits.BinTableHDU.from_columns(columns)
new_hdus = new_hdus.data[new_hdus.data['ra'] != 0]
#fits.BinTableHDU(new_hdus).writeto('')
print('Done')
