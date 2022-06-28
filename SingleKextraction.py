#%%
import numpy as np
import glob
import pickle
from skimage import io

# fonctions
from fcd import *
# from fcdgld import fcd_gld_hstar_series
from demodulation import demodulation_gld_sp
from save_deformation_field import save_deformation_field

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

#%%
folder = folderlist[0]
image_path = folder + '/image_sequence' # folder of the image sequence
d = {} # initialisation of the dictionnary 
files = glob.glob(image_path+'/*.tiff') # images 
print(folder, len(files))
save_path = folder + '/resultats' # save folder
    if os.path.isdir(save_path) == False:
         os.mkdir(save_path)
pdebut = ['facq','texp','Apot','fpot','Hw','pasdamier','Emembrane','diskE','R']
pfin = ['Hz','mus','mV','Hz','cm','mm','mm','mm','cm']
for (pd,pf) in zip(pdebut,pfin):
d[pd] = float(save_path.split('_'+pd)[1].split(pf+'_')[0])
facq = d['facq']
fpot = d['fpot']
diskE = d['diskE']
print('fpot = ', fpot, 'Edisque', diskE)       


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
eta = fcd_gld_hstar_series(idef_collection, carriers,alpha,hp,H, Nmax)
[nx, ny, nt] = eta.shape
X = np.arange(0,nx)*fx
Y = np.arange(0,ny)*fx
T = np.arange(nt)/facq
print('done')
print('Computing demdulation of the field....')
c, tf, etademod = demodulation_gld_sp(T,eta, fpot)
print('Saving....')
#save_deformation_field(c, save_path, filename = "\champ_deformation_detrend_demodulated_hxy_complexe_moyenne_sur_t")



# %%
