import numpy as np
from astropy.io import fits
from nbodykit.lab import *
from scipy import interpolate

def survey2cartesian(catalog_file,cosmology,minz_cut,catalog_type):
    if catalog_type == 'data':
        raw_data = FITSCatalog(catalog_file)
        raw_data = raw_data[raw_data['z']>minz_cut]
        cartesian_positions = (transform.SkyToCartesian(raw_data['ra'],raw_data['dec'],raw_data['z'],cosmo=cosmology)).compute().T
        density = raw_data.compute(raw_data['weight_sdc'])*(raw_data.compute(raw_data['weight_cp'])+raw_data.compute(raw_data['weight_noz'])-1.)*raw_data.compute(raw_data['weight_fkp'])
        #density = np.ones(len(raw_data))
    if catalog_type == 'randoms':
        raw_data = FITSCatalog(catalog_file)
        raw_data = raw_data[raw_data['z']>minz_cut]
        cartesian_positions = (transform.SkyToCartesian(raw_data['ra'],raw_data['dec'],raw_data['z'],cosmo=cosmology)).compute().T
        density = np.ones(len(raw_data))
    return(cartesian_positions,density)

def get_grid_index(cartesian_position,boxmin,boxsize,gridNumber):
    catalog_index = cartesian_position.copy()
    for i in range(len(catalog_index)):
        catalog_index[i] = np.floor((cartesian_position[i]-boxmin)*(gridNumber-1)/boxsize)
    return(catalog_index)

def grid_assignment(catalog_indices,gridNumber,density):
    axis = np.array([0. for i in range(gridNumber)])
    catalog_field = np.array(np.meshgrid(axis,axis,axis)[0])
    for i in range(len(catalog_indices.T)):
        if (catalog_indices[0,i] >= 0 and catalog_indices[0,i]<gridNumber) and (catalog_indices[1,i] >= 0 and catalog_indices[1,i]<gridNumber) and (catalog_indices[2,i] >= 0 and catalog_indices[2,i]<gridNumber):
            catalog_field[int(catalog_indices[0,i]),int(catalog_indices[1,i]),int(catalog_indices[2,i])] = catalog_field[int(catalog_indices[0,i]),int(catalog_indices[1,i]),int(catalog_indices[2,i])] + density[i]
    return(catalog_field)

def get_overdensity(field):
    mean = field[np.where(field!=0)].mean()
    overdensity = field.copy()
    overdensity[np.where(field==0)] = mean
    overdensity = (overdensity-mean)/mean
    return(overdensity)

def getWavenumber(gridNumber,boxsize):
    axis = np.array([0. for i in range(gridNumber)])
    wavenumber = np.array(np.meshgrid(axis,axis,axis)[0])
    dk = 2*np.pi/boxsize
    Axis = np.arange(gridNumber,dtype =float)
    mesh = np.array(np.meshgrid(Axis,Axis,Axis))
    xindex,yindex,zindex = mesh[0],mesh[1],mesh[2]
    wavenumber =dk*(xindex+yindex+zindex)
    wavenumber[wavenumber==0]=1
    return(wavenumber)

def shiftParticle(original,displacement,gridNumber,boxsize,boxmin):
    boxRange = np.arange(gridNumber)
    boxGrid = np.array([boxRange for i in range(3)])
    new = original.copy()
    new[0] = (original[0]-boxmin)*(gridNumber-1)/boxsize
    new[1] = (original[1]-boxmin)*(gridNumber-1)/boxsize
    new[2] = (original[2]-boxmin)*(gridNumber-1)/boxsize
    xinterp = interpolate.interpn(boxGrid,displacement[0],new.T,bounds_error=False,fill_value = 0).real
    yinterp = interpolate.interpn(boxGrid,displacement[1],new.T,bounds_error=False,fill_value = 0).real
    zinterp = interpolate.interpn(boxGrid,displacement[2],new.T,bounds_error=False,fill_value = 0).real
    new[0] = (new[0])*(boxsize)/(gridNumber-1)+boxmin+xinterp
    new[1] = (new[1])*(boxsize)/(gridNumber-1)+boxmin+yinterp
    new[2] = (new[2])*(boxsize)/(gridNumber-1)+boxmin+zinterp
    new = new.transpose()
    return(new)
