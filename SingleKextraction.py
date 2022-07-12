#%%
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from skimage import io
from tools import cart2pol
from display import set_size, vcolors
vcolor = vcolors(n=7)
# fonctions fcd
from fcd import *
from demodulation import demodulation
from save_deformation_field import save_deformation_field

# kextraction
from accum import accum
from radialavg2 import radialavg2
import matplotlib as mpl
import matplotlib.pyplot as plt
from kextraction import kextraction

#%%
date = '09062022'
root = 'X:/GLD/Banquise/Data/'
Directory = root+'d' + date
print(Directory+'/d'+date+'*')
folderlist = glob.glob(Directory+'/d'+date+'*')
print(folderlist)
exec(open(Directory + '/Parametres_jour.py').read())

#%%
Nmax = 530
hp = hpipx # distance pattern interface en pixels
H = Hiopx  # ditance interface objectif en pixels
facq = 26.5
fpot = 150
#%%
folder = folderlist[4]
image_path = folder + '/image_sequence' # folder of the image sequence
d = {} # initialisation of the dictionnary 

files = glob.glob(image_path+'/*.tiff') # images 
print(folder, len(files))
save_path = folder + '/resultats' # save folder

if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
#%%

# Reference image
iref = io.imread_collection(os.path.join(image_path, "Image_ref*.tiff"),plugin = "tifffile", conserve_memory=True)[0]; # image de reference
print(f'processing reference image...', end='')
carriers = calculate_carriers(iref)
print('done')

# Deformed images
idef_collection = io.imread_collection(os.path.join(image_path, "Basler*.tiff"),plugin = "tifffile", conserve_memory=True) #liste des images 
print('On trouve ...' + str(len(idef_collection)) + ' images')
print('Computing fast checkerboard demodulation....')
Nmax = 50
eta = fcd_hstar_series(idef_collection, carriers,alpha,hp,H, Nmax)
[nx, ny, nt] = eta.shape
X = np.arange(0,nx)*fx
Y = np.arange(0,ny)*fx
T = np.arange(nt)/facq
print('done')
print('Computing demdulation of the field....')
c = demodulation(T,eta, fpot)
print('Saving....')
#save_deformation_field(c, save_path, filename = "\champ_deformation_detrend_demodulated_hxy_complexe_moyenne_sur_t")

#%%

# if the demodulated field exist already
from scipy.io import loadmat
c = loadmat(save_path + '/champ_demodule.mat')['output']

#%%
fig, ax =plt.subplots(1,1, figsize = set_size(width = 400, subplots = (1,1)))
im = ax.pcolormesh(np.real(c)*fx)
#plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#ax.set_xlabel(r'$x$ (cm)')
#ax.set_ylabel(r'$y$ (cm)')
#cbar = fig.colorbar(im, format='%.0e', fraction=0.0375, pad=0.02)
cbar = fig.colorbar(im, fraction=0.0375, pad=0.02)

#cbar.set_label(r'$\eta(x,y)$ (cm)') 
# plt.ylim([0, 26])
# plt.xlim([0, 33])
ax.set_aspect('equal', 'box')
plt.tight_layout()
#plt.savefig('./docs/source/images/Champ_deformation_n40.png', dpi = 300)

#%% 
# Compute the k field
step_ana = 1
fitlength = 3
from radialavg2 import radialavg2

def kextraction(data, fitlength, step_ana):
    """
    Extracts the wavenum oneach point of the wavefield.
    It calls for the function radialavg2 to reconstruct the Bessel function of first order on each ooint of the 2D matrix data.
    :param: 
   
        * data : complex demodulated field;
        * fitelength : resolution of the reconstructed bessel function;
        * step_ana : step of analysis.
   
    :return:
   
        Return k the wavenum field
    
    Example
    -------
    >>> step_ana = 1
    >>> fitlength = 30
    >>> kfield  = kextraction(c, fitlength, step_ana)
    """

    [nx,ny] = data.shape
    cx = 0 
    k2 = np.zeros((int((ny-fitlength)/step_ana), int((nx-fitlength)/step_ana)-1))
    phase_locale = np.ones((2*fitlength,2*fitlength))
    signal_local = np.zeros(phase_locale.shape)
    for x0 in range(fitlength, nx-fitlength+1, step_ana):
        if np.mod(x0,60)==0:
            print(str(np.round(x0*100/(nx-fitlength),0))+ ' % ')
        cy = 0
        for y0 in range(fitlength, ny-fitlength+1, step_ana):
            phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(data[x0,y0]))
            signal_local = np.real(data[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)
            [r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1)
            xx = r2[0:fitlength]
            xx2 = np.concatenate((np.flipud(-xx),xx))
            test = np.abs(zr2[0:fitlength])
            test2 = np.concatenate((np.flipud(test),test))
            pp = np.polyfit(xx2,test2,deg=2)
            pp[0]=np.abs(pp[0])
            pp[2]=np.abs(pp[2])
            k2[cy,cx]=np.sqrt(4*pp[0]/pp[2])
            cy+=1
        cx += 1
    return k2

kfield  = kextraction(c, fitlength, step_ana)
print('done')
#%%
fig, ax = plt.subplots(1,1, figsize = set_size(width = 400, subplots = (1,1)))
ax.pcolormesh(kfield)

