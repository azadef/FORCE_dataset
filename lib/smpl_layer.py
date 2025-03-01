import os

import numpy as np
import torch
from torch.nn import Module

from smplpytorch.native.webuser.serialization import ready_arguments
from smplpytorch.pytorch.tensutils import th_posemap_axisang, th_with_zeros, th_pack, make_list
from scipy.sparse import csc_matrix

class SMPL_Layer(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                 normalise=False,
                 center_idx=None,
                 gender='neutral',
                 model_root='./lib/', num_betas=10, hands=False):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super(SMPL_Layer, self).__init__()
        self.normalise = normalise
        self.center_idx = center_idx
        self.gender = gender
        self.hands = hands
        if gender == 'neutral':
            #SMPL + H
            if self.hands:
                self.model_path = dict(
                    np.load(os.path.abspath('lib/neutral.npz'), allow_pickle=True))
                print(self.model_path.keys())
                self.model_path['bs_type'] = self.model_path['bs_type'].item().decode('utf-8')
                self.model_path['J_regressor'] = csc_matrix(self.model_path['J_regressor'])

            else:
                self.model_path = os.path.abspath('SMPL_models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

        elif gender == 'female':
            # self.model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
            if self.hands:
                self.model_path = os.path.join('SMPL_models/SMPLH_female.pkl')
            else:
                self.model_path = os.path.abspath(os.path.join(model_root, 'female_model.pkl'))

        elif gender == 'male':
            # self.model_path = os.path.join(model_root, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')
            # self.model_path = os.path.join(model_root, 'male_model.pkl')
            if self.hands:
                self.model_path = os.path.join('SMPL_models/SMPLH_male.pkl')
            else:
                self.model_path = os.path.abspath('SMPL_models/basicModel_male_lbs_10_207_0_v1.0.0.pkl')

        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'][:, :, :num_betas].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())
        # self.pose_subjects = smpl_data['pose_subjects']

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24

    # def forward(self,
    #             pose,
    #             betas=torch.zeros(1),
    #             trans=torch.zeros(1, 3),
    #             scale = 1.):
    #     """
    #     Args:
    #     th_pose_axisang (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    #     th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
    #     th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
    #     th_offsets (Tensor (batch_size x 6890 x 3)): if provided, adds per-vertex offsets in t-pose
    #     """
    #     batch_size = pose.shape[0]
    #
    #     # Convert axis-angle representation to rotation matrix rep.
    #     th_pose_rotmat = th_posemap_axisang(pose)
    #     # Take out the first rotmat (global rotation)
    #     root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
    #     # Take out the remaining rotmats (23 joints)
    #     th_pose_rotmat = th_pose_rotmat[:, 9:]
    #
    #     # else:
    #     th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)
    #     th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
    #     # if self.normalise:
    #     #     th_j = th_j - th_j[:, 0:1].detach().clone()
    #     # Global rigid transformation
    #     th_results = []
    #     root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
    #     th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))
    #
    #     # Rotate each part
    #     for i in range(self.num_joints - 1):
    #         i_val = int(i + 1)
    #         joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
    #                                9].contiguous().view(batch_size, 3, 3)
    #         joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
    #         parent = make_list(self.kintree_parents)[i_val]
    #         parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
    #         joint_rel_transform = th_with_zeros(
    #             torch.cat([joint_rot, joint_j - parent_j], 2))
    #         th_results.append(
    #             torch.matmul(th_results[parent], joint_rel_transform))
    #     th_results_global = th_results
    #
    #     th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]
    #
    #     # Scale
    #     th_jtr = (th_jtr ) * scale
    #     if self.normalise:
    #         th_jtr = th_jtr - th_jtr[:, 0:1].detach().clone()
    #
    #     th_jtr = th_jtr + trans.unsqueeze(1)
    #     # print(th_jtr.mean())
    #     # Vertices and joints in meters
    #     return th_jtr, torch.stack(th_results_global, 1)

    def subtract_flat_id(self, rot_mats, hands=True):
        # Subtracts identity as a flattened tensor
        if hands:
            num_joints = 51
        else:
            num_joints = 23
        id_flat = torch.eye(
            3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
            rot_mats.shape[0], num_joints)  # 23
        # id_flat.requires_grad = False
        results = rot_mats - id_flat
        return results

    def forward(self,
                pose,
                betas=torch.zeros(1),
                trans=torch.zeros(1, 3),
                th_offsets=None, scale=1.):
        """
        Args:
        th_pose_axisang (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        th_offsets (Tensor (batch_size x 6890 x 3)): if provided, adds per-vertex offsets in t-pose
        """
        batch_size = pose.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(pose)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = self.subtract_flat_id(th_pose_rotmat, self.hands) #

        # Below does: v_shaped = v_template + shapedirs * betas
        # If shape parameters are not provided
        # if th_betas is None or bool(torch.norm(th_betas) == 0):
        # th_v_shaped = self.th_v_template + torch.matmul(
        #     self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
        # th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
        #     batch_size, 1, 1)
        # else:
        th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)
        th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        # Below does: v_posed = v_shaped + posedirs * pose_map
        naked = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        if th_offsets is not None:
            th_v_posed = naked + th_offsets
        else:
            th_v_posed = naked
        # Final T pose with transformation done!

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                                          9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)
        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # Scale
        # th_verts1 = (th_verts - th_jtr[:, 0].unsqueeze(1)) * scale + th_jtr[:, 0].unsqueeze(1)
        # th_jtr1= (th_jtr - th_jtr[:, 0].unsqueeze(1)) * scale + th_jtr[:, 0].unsqueeze(1)

        if self.normalise:
            th_verts = th_verts - th_jtr[:, 0:1].detach().clone()

            th_jtr = th_jtr - th_jtr[:, 0:1].detach().clone()

        th_verts = (th_verts) * scale
        th_jtr = (th_jtr) * scale

        # If translation is not provided
        # if th_trans is None or bool(torch.norm(th_trans) == 0):
        #     if self.center_idx is not None:
        #         center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
        #         th_jtr = th_jtr - center_joint
        #         th_verts = th_verts - center_joint
        # else:
        th_jtr = th_jtr + trans.unsqueeze(1)
        th_verts = th_verts + trans.unsqueeze(1)

        # Vertices and joints in meters
        return th_verts, th_jtr, th_v_posed, naked, torch.stack(th_results_global, 1)


def global_rigid_transformation(J, th_pose_axisang, th_trans, rents):
    batch_size = th_pose_axisang.shape[0]
    # Convert axis-angle representation to rotation matrix rep.
    th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
    # Take out the first rotmat (global rotation)
    root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
    th_j = J.reshape(1, -1, 3).repeat(batch_size, 1, 1)

    th_results = []

    root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
    th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

    # Rotate each part
    for i in range(th_j.shape[1] - 1):
        i_val = int(i + 1)
        joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                                      9].contiguous().view(batch_size, 3, 3)
        joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
        parent = make_list(rents)[i_val]
        parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
        joint_rel_transform = th_with_zeros(
            torch.cat([joint_rot, joint_j - parent_j], 2))
        th_results.append(
            torch.matmul(th_results[parent], joint_rel_transform))
    return torch.stack(th_results, dim=1)[:, :, :3, 3]
