import numpy as np
def save_deformation_field(eta,save_path, filename = "champ_deformation_hxyt"):
    print(f'Saving deformation field...', end='')
    np.save(save_path + filename + ".npy",eta)
    print(f'Done')