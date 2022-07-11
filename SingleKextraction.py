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


from scipy.io import loadmat
c = loadmat(save_path + '/champ_demodule.mat')['output']

#%%
fig, ax =plt.subplots(1,1, figsize = set_size(width = 400, subplots = (1,1)))
im = ax.pcolormesh(np.real(c))
#plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.set_xlabel(r'$x$ (cm)')
ax.set_ylabel(r'$y$ (cm)')
cbar = fig.colorbar(im, format='%.0e', fraction=0.0375, pad=0.02)
cbar.set_label(r'$\eta(x,y)$ (cm)') 
# plt.ylim([0, 26])
# plt.xlim([0, 33])
ax.set_aspect('equal', 'box')
plt.tight_layout()
#plt.savefig('./docs/source/images/Champ_deformation_n40.png', dpi = 300)

#%%
pas = 1
step_ana = 1
fitlength = 30
kfield  = kextraction(c, fitlength, step_ana, step)

print('done')

 #%%

fig, ax = plt.subplots(1,1, figsize = set_size(width = 400, subplots = (1,1)))
ax.pcolomesh(k2)


 #%%
# def radialavg2(data, 1, radial_step, x0,y0, m):
#     l = np.int(data.size/2)
 # %%
