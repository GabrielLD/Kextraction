#%%
import numpy as np
import glob
import pickle
from skimage import io
from tools import cart2pol

# fonctions
from fcd import *
# from fcdgld import fcd_gld_hstar_series
from demodulation import demodulation_gld_sp
from save_deformation_field import save_deformation_field

#%%
date = '02052022'
root = 'X:/Mina/Data/'
Directory = root+'d' + date
print(Directory+'/d'+date+'*')
folderlist = glob.glob(Directory+'/d'+date+'*')
print(folderlist)
exec(open(Directory + '/Parametres_jour.py').read())

#%%
Nmax = 530
hp = hpipx # distance pattern interface en pixels
H = Hiopx  # ditance interface objectif en pixels
facq = 51
fpot = 20
#%%
folder = folderlist[7]
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
c, tf, etademod = demodulation_gld_sp(T,eta, fpot)
print('Saving....')
#save_deformation_field(c, save_path, filename = "\champ_deformation_detrend_demodulated_hxy_complexe_moyenne_sur_t")

#%%


from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
c = loadmat(save_path + '/champ_demodule.mat')['output']

fig, ax =plt.subplots(1,1)
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
# %%

from accum import accum

x0 = 200
y0 = 300

phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(c[x0,y0]))
signal_local=np.zeros(phase_locale.shape)
signal_local[:,:] = np.real(c[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)

def radialavg2(data, radial_step,x0,y0,m):
    l = np.int(data.shape[0]/2)
    x = np.arange(0,data.shape[1]) - data.shape[1]/2+(l-x0)
    y = np.arange(0,data.shape[0]) - data.shape[0]/2+(l-y0)
    [X,Y] = np.meshgrid(x,y)
    [R, Theta] = cart2pol(X,Y)
    Zinteger = np.zeros(R.shape)
    Zinteger = R/radial_step
    Zinteger = np.abs(X+1j*Y)/radial_step+1
    Tics = accum(Zinteger.astype(int), np.abs(X+1j*Y), func = np.mean)
    Average = accum(Zinteger.astype(int), data, func = np.mean)
    return Tics, Average

[r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1, m)
plt.plot(r2,zr2)

#%%
[nx,ny] = c.shape
cx = 0 
pas = 1
k2 = np.zeros((int((ny/fitlength)/step_ana), int((nx-fitlength)/step_ana)-1))
step_ana = 1
fitlength = 6
m= 0
phase_locale = np.ones((2*fitlength,2*fitlength))
signal_local = np.zeros(phase_locale.shape)
for x0 in range(fitlength, nx-fitlength+1, step_ana):
    if np.mod(x0,60)==0:
        print(str(np.round(x0*100/(nx-fitlength),0))+ ' % ')
    cy = 0
    for y0 in range(fitlength, ny-fitlength+1, step_ana):
        phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(c[x0,y0]))
        signal_local = np.real(c[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)
        [r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1, m)
        xx = r2[0:fitlength]*pas
        xx2 = np.concatenate((np.flipud(-xx),xx))
        test = np.abs(zr2[0:fitlength])
        test2 = np.concatenate((np.flipud(test),test))
        pp = np.polyfit(xx2,test2,deg=2)
        pp[0]=np.abs(pp[0])
        pp[2]=np.abs(pp[2])
        k2[cy,cx]=np.sqrt(4*pp[0]/pp[2])
        cy=+1
    cx =+ 1
print('done')

 #%%

fig, ax = plt.subplots(1,1, figsize = (15,8))#set_size(width = 400, subplots = (1,1)))
ax.pcolomesh(k2)


 #%%
# def radialavg2(data, 1, radial_step, x0,y0, m):
#     l = np.int(data.size/2)
 # %%
