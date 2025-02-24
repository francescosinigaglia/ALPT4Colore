import numpy as np
from numba import njit, prange
import os
import time
import astropy.io.fits as fits
import healpy
import astropy.constants as const
from astropy.table import Table
from multiprocessing import Pool, cpu_count
import input_params as inpars

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# I/O files           
nreal = 3#inpars.nreal
version = 'v8.8'#inpars.version

# Input filenames     
input_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_boxes/v0/box%d/' %nreal
output_aux_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/aux/' %nreal
output_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/' %nreal

output_colore = '/global/cfs/cdirs/desi/users/fsin/ALPT4Colore/'

# General parameters
posx_qso_filename = output_aux_dir + 'posxtr_zspace.dat'
posy_qso_filename = output_aux_dir + 'posytr_zspace.dat'
posz_qso_filename = output_aux_dir + 'posztr_zspace.dat'

fits_filename = output_dir + 'master.fits'

delta_filename = input_dir + 'dmLCOM0.314OL0.686G1800V10000.0.dat'

vx_filename = input_dir + 'VEULxOM0.314OL0.686G1800V10000.0.dat'
vy_filename = input_dir + 'VEULyOM0.314OL0.686G1800V10000.0.dat'
vz_filename = input_dir + 'VEULzOM0.314OL0.686G1800V10000.0.dat'

zarr_filename = input_dir + 'zarr.dat'
darr_filename = input_dir + 'dcomOM0.314OL0.686.dat'

nside = 16
pixmax = 3072

num_processes = 1#96

# General parameters
lbox = 10000.
ngrid = 1800 

zmin = 1.77
zmax = 3.8

zmin_extr = 1.77#2.

hrbinw = 2.44140 #0.1

lammin = 3469.9
lammax = 6500.1
dlam = 0.2

# Rest-frame frequencies
lam_lya = 1215.67
lam_lyb = 1025.72
lam_SiII_1260 = 1260.42
lam_SiIII_1207 = 1206.50
lam_SiII_1193 = 1193.29
lam_SiII_1190 = 1190.42

# Observer positions
obspos = [lbox/2., lbox/2., lbox/2.]

# Cosmological parameters (Abacus)
h = 0.6736
H0 = 100
Om = 0.314
Orad = 0.
Ok = 0.
N_eff = 3.046
w_eos = -1
Ol = 1-Om-Ok-Orad

# Random seed for stochasticity reproducibility
np.random.seed(123456)

smallnum = 1e-6

# **********************************************
# **********************************************
# **********************************************
# **********************************************                                                                            
# **********************************************
# **********************************************
def extract_skewers(posx, posy, posz, zmin, zmax, zarr, darr, hrbinw, delta, vx, vy, vz, ngrid, lbox, xobs, yobs, zobs):

    lcell = lbox / ngrid

    # Initialize a matrix of skewers                                                                                                                            
    # (NxM), where N=number of QSO, M=number of bins per spectrum                                                                                               

    # Let's first determine maximum and minimum distance, given the redshift range                                                                              
    dmax = np.interp(zmax, zarr, darr)
    dmin = np.interp(zmin, zarr, darr)

    # Determine the number of bins in the spectra                                                                                                               
    nbins = int((dmax-dmin) / hrbinw)

    # Allocate the matrix                                                                                                                                       
    skmat = np.zeros((len(posx), nbins))
    velmat = np.zeros((len(posx), nbins))

    # Set up a template of distances                                                                                                                            
    dtemplate = np.linspace(dmin, dmax, nbins+1)
    dtemplate = 0.5*(dtemplate[1:] + dtemplate[:-1])

    ztemplate = np.interp(dtemplate, darr, zarr)

    for ii in range(len(posx)):

        ra, dec, zz = cartesian_to_sky(posx[ii],posy[ii],posz[ii], zarr, darr, xobs, yobs, zobs)
        #print(zz)

        for jj in range(len(dtemplate)):

            ztemp = ztemplate[jj]

            posxlya, posylya, poszlya = sky_to_cartesian(ra,dec,ztemp, zarr, darr, xobs, yobs, zobs)
            indx = int(posxlya/lcell)
            indy = int(posylya/lcell)
            indz = int(poszlya/lcell)

            skmat[ii,jj] = delta[indx,indy,indz]
            vxlya = vx[indx,indy,indz]
            vylya = vy[indx,indy,indz]
            vzlya = vz[indx,indy,indz]
            vx_proj, vy_proj, vz_proj = project_vector_los(posxlya, posylya, poszlya, vxlya, vylya, vzlya, zarr, darr, xobs, yobs, zobs)
            velmat[ii,jj] = np.sqrt(vx_proj**2 + vy_proj**2 + vz_proj**2)

    return ztemplate, dtemplate, skmat, velmat


# **********************************************
@njit(parallel=False, fastmath=True, cache=True)
def sky_to_cartesian(ra,dec,zz, zarr, darr, xobs, yobs, zobs):

    dd = np.interp(zz, zarr, darr)

    ra = ra / 180. * np.pi
    dec = dec / 180. * np.pi

    posx = dd * np.cos(dec) * np.cos(ra)
    posy = dd * np.cos(dec) * np.sin(ra)
    posz = dd * np.sin(dec)

    posx += xobs
    posy += yobs
    posz += zobs

    return posx, posy, posz

# **********************************************
@njit(parallel=False, fastmath=True, cache=True)
def cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs):

    posx -= xobs
    posy -= yobs
    posz -= zobs
    
    dd = np.sqrt(posx**2 + posy**2 + posz**2)
    zz = np.interp(dd, darr, zarr)

    ss = np.hypot(posx, posy)
    ra = np.arctan2(posy, posx) / np.pi * 180.
    dec = np.arctan2(posz, ss) / np.pi * 180.

    return ra, dec, zz

# **********************************************

@njit(parallel=False, fastmath=True, cache=True)
def project_vector_los(posx, posy, posz, vecx, vecy, vecz, zarr, darr, xobs, yobs, zobs):

    # Determine the line of sight angles ra0 and dec0
    ra0, dec0, zz0 = cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs)

    ra0 = ra0 / 180. * np.pi
    dec0 = dec0 / 180. * np.pi
    
    # Find the l.o.s. unit vector
    versx = np.cos(dec0) * np.cos(ra0)
    versy = np.cos(dec0) * np.sin(ra0)
    versz = np.sin(dec0)
    
    # Project the velocity vector along the l.o.s. direction
    norm = vecx*versx + vecy*versy + vecz*versz
    
    vecx = norm * versx
    vecy = norm * versy
    vecz = norm * versz

    return vecx, vecy, vecz

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):

    lcell = lbox/ngrid

    indxc = int(xx/lcell)
    indyc = int(yy/lcell)
    indzc = int(zz/lcell)

    wxc = xx/lcell - indxc
    wyc = yy/lcell - indyc
    wzc = zz/lcell - indzc

    if wxc <=0.5:
        indxl = indxc - 1
        if indxl<0:
            indxl += ngrid
        wxc += 0.5
        wxl = 1 - wxc
    elif wxc >0.5:
        indxl = indxc + 1
        if indxl>=ngrid:
            indxl -= ngrid
        wxl = 1 - wxc

    if wyc <=0.5:
        indyl = indyc - 1
        if indyl<0:
            indyl += ngrid
        wyc += 0.5
        wyl = 1 - wyc
    elif wyc >0.5:
        indyl = indyc + 1
        if indyl>=ngrid:
            indyl -= ngrid
        wyl = 1 - wyc

    if wzc <=0.5:
        indzl = indzc - 1
        if indzl<0:
            indzl += ngrid
        wzc += 0.5
        wzl = 1 - wzc
    elif wzc >0.5:
        indzl = indzc + 1
        if indzl>=0:
            indzl -= ngrid
        wzl = 1 - wzc

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    return out

# **********************************************
# **********************************************
# **********************************************
print('--------------------------------')
print('Extract and regrid Lya skewers')
print('--------------------------------')

ti = time.time()

lcell = lbox/ngrid

xobs = obspos[0]
yobs = obspos[1]
zobs = obspos[2]

# Read the tabulated redshift and comoving distance arrays                                                                                                      
zarr = np.fromfile(zarr_filename, dtype=np.float32)
darr = np.fromfile(darr_filename, dtype=np.float32)


print('Read QSO positions ...')
# Now read QSO positions in redshift space                                        
# The containers are used first for RA,DEC,z     
cx = np.fromfile(open(posx_qso_filename, 'r'), dtype=np.float32)
cy = np.fromfile(open(posy_qso_filename, 'r'), dtype=np.float32)
cz = np.fromfile(open(posz_qso_filename, 'r'), dtype=np.float32)
print('... done!')
print('') 

# Now open the fits catalog                                                                                                                                     
rawfits = fits.open(fits_filename)
catfits = rawfits[1].data
ra = catfits['RA']
dec = catfits['DEC']
zz = catfits['Z']
mockid = catfits['MOCKID']

# Read density and velocity field                                                                                                                                               
delta = np.fromfile(open(delta_filename, 'r'), dtype=np.float32)
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

vx = np.fromfile(open(vx_filename, 'r'), dtype=np.float32)
vy = np.fromfile(open(vy_filename, 'r'), dtype=np.float32)
vz = np.fromfile(open(vz_filename, 'r'), dtype=np.float32)
vx = np.reshape(vx, (ngrid,ngrid,ngrid))
vy = np.reshape(vy, (ngrid,ngrid,ngrid))
vz = np.reshape(vz, (ngrid,ngrid,ngrid))

# Cut the QSO positions - keep only QSOs relevant for lya                                                                                                       
cx = cx[np.logical_and(zz>zmin, zz<zmax)]
cy = cy[np.logical_and(zz>zmin, zz<zmax)]
cz = cz[np.logical_and(zz>zmin, zz<zmax)]

ra = ra[np.logical_and(zz>zmin, zz<zmax)]
dec = dec[np.logical_and(zz>zmin, zz<zmax)]
mockid = mockid[np.logical_and(zz>zmin, zz<zmax)]
zz = zz[np.logical_and(zz>zmin, zz<zmax)]

healpix = healpy.ang2pix(nside, np.radians(90.-dec), np.radians(ra), nest=True)

# Define the DESI footprint
footprint = fits.open('DESI_footprint_nside16.fits')
dark = footprint[1].data['DESI_DARK']

#print('Extracting skewers ...')

# ****************************************
def extract_and_regrid_parallel(ii):

    dirnum = int(ii/100)

    if dark[ii]==True:

        print(ii)

        #if not os.path.exists(output_dir + '/%d/%d/' %(dirnum,ii)):
        #    os.mkdir(output_dir + '%d/%d/' %(dirnum,ii))
            
        xx = cx[healpix==ii]
        yy = cy[healpix==ii]
        zz = cz[healpix==ii]
        mockidd = mockid[healpix==ii]

        ztemplate, dtemplate, densmat, velmat = extract_skewers(xx, yy, zz, zmin, zmax, zarr, darr, hrbinw, delta, vx, vy, vz, ngrid, lbox, xobs, yobs, zobs)
        
        # Now construct the fits file
        # Read QSO catalog and build HDU
        rawqso = fits.open(output_dir + 'master.fits')

        qsotab = rawqso[1].data
        arr, inddx, inddx2 = np.intersect1d(qsotab['MOCKID'], mockidd, return_indices=True)
        data = qsotab[inddx]

        typelyacolore = np.zeros(len(data['RA'])).astype('int')
        dz = data['Z']-data['Z_noRSD']

        c0 = fits.Column(name='TYPE', array=typelyacolore, format='1J', unit='NA')
        c1 = fits.Column(name='RA', array=data['RA'], format='1E', unit='DEG')
        c2 = fits.Column(name='DEC', array=data['DEC'], format='1E', unit='DEG')
        c3 = fits.Column(name='Z_COSMO', array=data['Z'], format='1E')
        c4 = fits.Column(name='DZ_RSD', array=dz, format='1E')
        hdu1 = fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4], name='CATALOG')

        # Read DLA catalog and build HDU

        # Generate two dummy arrays - D and V - are reasonabel but ill need to be replaced with the right ones
        DD = np.linspace(1., 0.25, num=len(dtemplate))
        VV = np.linspace(1., 2.94, num=len(dtemplate))

        c0 = fits.Column(name='R', array=dtemplate, format='1E', unit='MPC_H')
        c2 = fits.Column(name='Z', array=ztemplate, format='1E', unit='NA')
        c2 = fits.Column(name='D', array=DD, format='1E', unit='NA')
        c3 = fits.Column(name='V', array=VV, format='1E', unit='NA')
        hdu2 = fits.BinTableHDU.from_columns([c0, c1, c2, c3])

        hdu_list = fits.HDUList([
            fits.PrimaryHDU(),
            hdu1,
            fits.ImageHDU(densmat),
            fits.ImageHDU(velmat),
            hdu2])

        hdu_list[1].name = 'CATALOG'
        hdu_list[2].name = 'DENSITY'
        hdu_list[3].name = 'VELOCITY'

        #hdu_list[1].header['HPXNSIDE'] = 16
        #hdu_list[1].header['HPXPIXEL'] = ii
        #hdu_list[1].header['HPXNEST'] = True
        #hdu_list[1].header['LYA'] = 1215.67

        hdu_list.writeto(output_colore + 'alpt4colore_example.fits', overwrite=True)

        #break
        

# ******************************************************************

tin = time.time()

# First check if the master directory exists. If not, create it, together with the subdirectories
#if not os.path.exists('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version):
#    os.mkdir('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version)

#if not os.path.exists(output_dir):
#    os.mkdir(output_dir)

#for ii in range(pixmax):
#    if not os.path.exists(output_dir + '%d/' %(ii//100)):
#        os.mkdir(output_dir + '%d/' %(ii//100))

# Now start extraction and regridding
ii_list = [(ii) for ii in range(pixmax)]

if num_processes<0:
    num_processes = cpu_count()
else:
    num_processes = num_processes

with Pool(processes=num_processes) as pool:
    pool.map(extract_and_regrid_parallel, ii_list)

tfin = time.time()

dt = (tfin-tin)/60.

print('Elapsed ' + str(dt) + ' minutes ...')