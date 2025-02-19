import numpy as np
from os.path import join
from lib.smpl_layer import SMPL_Layer
from lib.meshviewer import Mesh, MeshViewer, colors, FloorMesh
from lib.rotation_conversions import *
import time

import pickle as pkl
from glob import glob

SCAN_PATH = {
    "backpack": "assets/backpack_closed_f1000.ply",
    'plasticcontainer': "assets/container_closed_f1000.ply",
    "suitcase": "assets/suitcase_flipped.ply",
    "laundrybasket": "assets/laundrybasket.ply",
    "chairblack_pushpull": "assets/chairblack_pushpull.ply",
    "box_huge": "assets/box_huge.ply"
}

seq_file = open('./annotations_final/seqs.txt')
seqs = seq_file.readlines()
obj_file = open('./annotations_final/objs.txt')
objs = obj_file.readlines()
seq_obj_dict = {}
for i, seq in enumerate(seqs):
    seq_obj_dict[seq.strip()] = objs[i].strip()

if __name__ == '__main__':

    mv = MeshViewer()
    smpl_hands = SMPL_Layer('male', hands=True)
    betas = torch.from_numpy(pkl.load(open('./assets/betas.pkl', 'rb'))['beta']).float().reshape(1, 10)
    data_paths = glob(join('./force_npz/*.npz'))
    for data_path in data_paths:
        seq_name = data_path.split('/')[-1].split('.npz')[0]
        if seq_obj_dict[seq_name] == '0':
            category = 'backpack'
        elif seq_obj_dict[seq_name] == '1':
            category = 'plasticcontainer'
        elif seq_obj_dict[seq_name] == '2':
            category = 'suitcase'
        elif seq_obj_dict[seq_name] == '3':
            category = 'laundrybasket'
        elif seq_obj_dict[seq_name] == '4':
            category = 'chairblack_pushpull'
        elif seq_obj_dict[seq_name] == '5':
            category = 'box_huge'

        # load obj mesh
        obj_mesh = Mesh(filename=SCAN_PATH[category])
        obj_mesh_centre = 0.5 * (obj_mesh.vertices.max(0) + obj_mesh.vertices.min(0))
        obj_mesh_vertices = obj_mesh.vertices - obj_mesh_centre

        # load smpl and obj motion
        data = np.load(data_path, allow_pickle=True)
        smpl_pose_params = data['smpl_pose']
        smpl_trans_params = data['smpl_trans']
        obj_pose_params = data['obj_pose']
        obj_trans_params = data['obj_trans']
        fit_pose_params = torch.from_numpy(smpl_pose_params).float()
        fit_trans_params = torch.from_numpy(smpl_trans_params).float()
        obj_pose_params = torch.from_numpy(obj_pose_params).float()
        obj_trans_params = torch.from_numpy(obj_trans_params).float()

        # smpl FK
        fit_verts = smpl_hands(pose=fit_pose_params.float(),
                               trans=fit_trans_params.float(),
                               betas=betas.repeat(fit_pose_params.shape[0], 1),
                               scale=1)[0].numpy()

        # visualise
        for i in range(fit_verts.shape[0]):
            obj_mesh.vertices = (torch.matmul(axis_angle_to_matrix(obj_pose_params[i]).float(),
                                              torch.from_numpy(obj_mesh_vertices).float().T) +
                                 obj_trans_params[i].reshape(3, 1)).T.numpy()

            mv.viewer.render_lock.acquire()
            mv.set_static_meshes([Mesh(vertices=fit_verts[i], faces=smpl_hands.faces.numpy(), smooth=True, vc=colors['lightblue']), obj_mesh])
            mv.viewer.render_lock.release()
            time.sleep(0.03)
            print(i, obj_mesh.vertices.shape)
        break
