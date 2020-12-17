from scipy.io import loadmat

import pickle as pkl
import numpy as np


def load_3DDFA_bfm(bfm_pkl_file):
  bfm_3DDFA = pkl.load(open(bfm_pkl_file, 'rb'))
  u = bfm_3DDFA.get('u').astype(np.float32)
  w_shp = bfm_3DDFA.get('w_shp').astype(np.float32)
  w_exp = bfm_3DDFA.get('w_exp').astype(np.float32)
  # Load 68-point indexes
  keypoints = bfm_3DDFA.get('keypoints').astype(np.long)
  # Extract parameter for 68-point system
  u_base = u[keypoints].reshape(-1, 1)
  w_shp_base = w_shp[keypoints]
  w_exp_base = w_exp[keypoints]

  return u_base, w_shp_base, w_exp_base


def load_origininal_bfm(bfm_shape_mat_file, bfm_exp_mat_file):
  bfm_shape_original_model = loadmat(bfm_shape_mat_file)
  bfm_exp_original_model = loadmat(bfm_exp_mat_file)

  keypoints = bfm_shape_original_model['keypoints']
  keypoints = keypoints.reshape((68,))

  new_keypoints = np.ndarray(shape=(204,), dtype=int)
  for i in range(68):
    new_keypoints[i * 3] = int((keypoints[i] - 1) * 3)
    new_keypoints[i * 3 + 1] = int((keypoints[i] - 1) * 3 + 1)
    new_keypoints[i * 3 + 2] = int((keypoints[i] - 1) * 3 + 2)

  u_base = bfm_shape_original_model['mu_shape'] + bfm_exp_original_model['mu_exp']
  u_base = u_base[new_keypoints]
  w_shp_base = bfm_shape_original_model['w'][new_keypoints]
  w_shp_base = w_shp_base[:, :40]
  w_exp_base = bfm_exp_original_model['w_exp'][new_keypoints]
  w_exp_base = w_exp_base[:, :10]

  return u_base, w_shp_base, w_exp_base


bfm = pkl.load(
    open('/home/innovplus/Dream/Projects/3DDFA_V2/configs/bfm_noneck_v3.pkl',
         'rb'))

if __name__ == "__main__":
  u_3DDFA, w_shp_3DDFA, w_exp_3DDFA = load_3DDFA_bfm(
      '/home/innovplus/Dream/Projects/3DDFA_V2/configs/bfm_noneck_v3.pkl')
  u_ref, w_shp_ref, w_exp_ref = load_origininal_bfm(
      '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Shape.mat',
      '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Exp.mat'
  )

  assert(u_3DDFA.all() == u_ref.all())
  assert(w_shp_3DDFA.all() == w_shp_ref.all())
  assert(w_exp_3DDFA.all() == w_exp_ref.all())
