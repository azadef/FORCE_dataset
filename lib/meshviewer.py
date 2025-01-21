# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#

import numpy as np
import torch
import trimesh
import pyrender
from pyrender.light import DirectionalLight
from pyrender.node import Node
from PIL import Image


# from .utils import euler
def euler(rots, order='xyz', units='deg'):
    rots = np.asarray(rots)
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis == 'x':
                r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
            if axis == 'y':
                r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
            if axis == 'z':
                r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats


class PointCloud(trimesh.PointCloud):

    def __init__(self, filename=None, points=None, colors=None, process=False, **kwargs):

        if filename is not None:
            pc = trimesh.load(filename, process=process)
            points = pc.vertices
            colors = pc.colors

        super(PointCloud, self).__init__(vertices=points, colors=colors, process=process)

        if colors is not None:
            self.set_colors(colors)

    def colors_like(self, color, array, ids):
        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_colors(self, colors, point_ids=None):
        all_ids = np.arange(self.vertices.shape[0])
        if point_ids is None:
            point_ids = all_ids

        point_ids = all_ids[point_ids]
        new_colors = self.colors_like(colors, self.colors, point_ids)
        self.colors[:] = new_colors


class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process=False,
                 visual=None,
                 wireframe=False,
                 smooth=False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process=process)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices * vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self, color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self, vc, vertex_ids=None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self, fc, face_ids=None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)


# class MyViewer(pyrender.Viewer):
from pyglet.window import key


class MyViewer(pyrender.Viewer):
    def __init__(self, *args, **kwargs):
        super(MyViewer, self).__init__(*args, **kwargs)
        self.registered_keys = kwargs.get('registered_keys', {})
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

    def on_key_press(self, symbol, modifiers):
        if symbol in self.registered_keys:
            self.registered_keys[symbol] = True

    def on_key_release(self, symbol, modifiers):
        if symbol in self.registered_keys:
            self.registered_keys[symbol] = False


class Lines:
    def __init__(self, vertices, indices, colors=None):
        self.vertices = vertices
        self.indices = indices
        self.colors = colors if colors is not None else np.array([[1.0, 0.0, 0.0, 1.0]] * len(vertices))
from lib.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion
import torch
class MeshViewer(object):

    def __init__(self,
                 width=1200,
                 height=800,
                 bg_color=[0.0, 0.0, 0.0, 1.0],
                 offscreen=False,
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.bg_color = bg_color
        self.offscreen = offscreen
        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=(0.3, 0.3, 0.3),
                                    name='scene')

        self.aspect_ratio = float(width) / height
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.aspect_ratio)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -2., 1.5])

        self.cam = pyrender.Node(name='camera', camera=pc, matrix=camera_pose)

        self.scene.add_node(self.cam)

        if self.offscreen:
            light = Node(light=DirectionalLight(color=np.ones(3), intensity=3.0),
                         matrix=camera_pose)
            self.scene.add_node(light)
            self.viewer = pyrender.OffscreenRenderer(width, height)
        else:
            self.viewer = MyViewer(self.scene,
                                   use_raymond_lighting=True,
                                   viewport_size=(width, height),
                                   cull_faces=False,
                                   run_in_thread=True,
                                   registered_keys=registered_keys)

        for i, node in enumerate(self.scene.get_nodes()):
            if node.name is None:
                node.name = 'Req%d' % i

    def get_camera_pose(self):
        """
        Get the current camera pose as position and quaternion
        Returns:
            position (np.ndarray): XYZ position (3,)
            quaternion (np.ndarray): WXYZ quaternion (4,)
            y_angle (float): Y-angle rotation in degrees
        """
        import math
        from scipy.spatial.transform import Rotation
        def fix_y_up_rotation(R):
            y = R[1]
            forward = R[2]  # Assuming Z is forward
            new_y = np.array([0, 1, 0])
            new_x = np.cross(new_y, forward)
            new_x = new_x / np.linalg.norm(new_x)  # normalize
            new_forward = np.cross(new_x, new_y)
            corrected_R = np.vstack([new_x, new_y, new_forward])
            return corrected_R

        if not self.offscreen:
            # Get the camera pose directly from the viewer
            camera_pose = self.viewer._camera_node.matrix
            position = camera_pose[:3, 3]
            rotation_matrix = camera_pose[:3, :3]
            # rotation_matrix = fix_y_up_rotation(rotation_matrix)
            # rotation_matrix = np.matmul(rotation_matrix, Rotation.from_euler('Y', math.pi).as_matrix().T)

            quaternion = matrix_to_quaternion(torch.FloatTensor(rotation_matrix)).numpy()
            # Get Euler angles and extract y angle
            rx, ry, rz = matrix_to_euler_angles(torch.FloatTensor(rotation_matrix), 'XYZ')
            y_angle = np.degrees(ry)  # Convert to degrees

            return position, quaternion, y_angle
        return None, None, None



    def pyrender_to_blender_quaternion(self, matrix):
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        # Swap Y and Z axes and invert the new Z-axis
        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        matrix = np.matmul(swap, np.matmul(matrix, np.linalg.inv(swap)))

        # Extract the 3x3 rotation matrix
        rotation_matrix = matrix[:3, :3]

        # Convert the rotation matrix to a quaternion
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()

        return quaternion

    def get_camera_y_angle(self):
        """
        Get just the y-angle (yaw) of the camera
        Returns:
            y_angle (float): Y-angle rotation in degrees
        """
        if not self.offscreen:
            camera_pose = self.viewer._camera_node.matrix
            rotation_matrix = camera_pose[:3, :3]
            _, ry, _ = matrix_to_euler_angles(torch.FloatTensor(rotation_matrix), 'XYZ')
            return np.degrees(ry)
        return None

    def get_camera_euler_angles(self):
        """
        Get the current camera orientation in Euler angles (xyz convention)
        Returns:
            angles (tuple): (rx, ry, rz) in radians
        """
        if not self.offscreen:
            camera_pose = self.viewer._camera_node.matrix
            rotation_matrix = camera_pose[:3, :3]
            angles = matrix_to_euler_angles(torch.FloatTensor(rotation_matrix), 'XYZ')
            return angles
        return None

    def get_full_camera_info(self):
        """
        Get all camera information in a dictionary
        Returns:
            dict: Dictionary containing position, quaternion, euler angles, and y_angle
        """
        if not self.offscreen:
            position, quaternion, y_angle = self.get_camera_pose()
            euler_angles = self.get_camera_euler_angles()

            return {
                'position': position,
                'quaternion': quaternion,
                'euler_angles': {
                    'x': euler_angles[0],
                    'y': euler_angles[1],
                    'z': euler_angles[2]
                },
                'y_angle': y_angle
            }
        return None

    def is_key_pressed(self, key):
        return self.viewer.keys[key]

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_background_color(self, bg_color=[1., 1., 1.]):
        self.scene.bg_color = bg_color

    def to_pymesh(self, mesh):

        wireframe = mesh.wireframe if hasattr(mesh, 'wireframe') else False
        smooth = mesh.smooth if hasattr(mesh, 'smooth') else False
        return pyrender.Mesh.from_trimesh(mesh, wireframe=wireframe, smooth=smooth)

    def update_camera_pose(self, pose):
        if self.offscreen:
            self.scene.set_pose(self.cam, pose=pose)
        else:
            self.viewer._default_camera_pose[:] = pose

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def set_meshes(self, meshes=[], set_type='static'):

        if not self.offscreen:
            self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'static' in set_type and 'mesh' in node.name:
                self.scene.remove_node(node)
            elif 'dynamic' in node.name:
                self.scene.remove_node(node)

        for i, mesh in enumerate(meshes):
            mesh = self.to_pymesh(mesh)
            self.scene.add(mesh, name='%s_mesh_%d' % (set_type, i))

        if not self.offscreen:
            self.viewer.render_lock.release()

    def set_static_meshes(self, meshes=[]):
        self.set_meshes(meshes=meshes, set_type='static')

    def set_dynamic_meshes(self, meshes=[]):
        self.set_meshes(meshes=meshes, set_type='dynamic')

    def save_snapshot(self, save_path):
        if not self.offscreen:
            print('We do not support rendering in Interactive mode!')
            return
        color, depth = self.viewer.render(self.scene)
        img = Image.fromarray(color)
        img.save(save_path)

    def set_lines(self, lines, set_type='static'):
        if not self.offscreen:
            self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'static' in set_type and 'line' in node.name:
                self.scene.remove_node(node)
            elif 'dynamic' in node.name:
                self.scene.remove_node(node)

        for i, line in enumerate(lines):
            path_vertices = line.vertices
            path_edges = line.indices
            # Create a pyrender primitive for the lines
            primitive = pyrender.Primitive(positions=path_vertices, indices=path_edges, color_0=line.colors,
                                           mode=pyrender.constants.GLTF.LINES)

            # Create a pyrender mesh from the primitive
            mesh = pyrender.Mesh(primitives=[primitive])
            node = pyrender.Node(mesh=mesh, name='%s_line_%d' % (set_type, i))
            self.scene.add_node(node)

        if not self.offscreen:
            self.viewer.render_lock.release()

    def set_static_lines(self, lines=[]):
        self.set_lines(lines=lines, set_type='static')

    def set_dynamic_lines(self, lines=[]):
        self.set_lines(lines=lines, set_type='dynamic')


# def Spheres(points, radius=.001, vc=(0., 0., 1.), count=(5, 5)):
#     points = points.reshape(-1, 3)
#     n_points = points.shape[0]
#
#     spheres = []
#     for p in range(n_points):
#         sphs = trimesh.creation.uv_sphere(radius=radius, count=count)
#         sphs.apply_translation(points[p])
#         sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)
#
#         spheres.append(sphs)
#
#     spheres = Mesh.concatenate_meshes(spheres)
#     return spheres
class Spheres(object):
    def __init__(self, points, radius=0.001, vc=(0., 0., 1.)):
        self.points = points.reshape(-1, 3)
        self.radius = radius
        self.vc = vc

    def __str__(self):
        return f"Spheres: {self.points.shape[0]} spheres, radius {self.radius}"

    def to_mesh(self):
        v = np.array([[0.0000, -1.000, 0.0000], [0.7236, -0.447, 0.5257],
                      [-0.278, -0.447, 0.8506], [-0.894, -0.447, 0.0000],
                      [-0.278, -0.447, -0.850], [0.7236, -0.447, -0.525],
                      [0.2765, 0.4472, 0.8506], [-0.723, 0.4472, 0.5257],
                      [-0.720, 0.4472, -0.525], [0.2763, 0.4472, -0.850],
                      [0.8945, 0.4472, 0.0000], [0.0000, 1.0000, 0.0000],
                      [-0.165, -0.850, 0.4999], [0.4253, -0.850, 0.3090],
                      [0.2629, -0.525, 0.8090], [0.4253, -0.850, -0.309],
                      [0.8508, -0.525, 0.0000], [-0.525, -0.850, 0.0000],
                      [-0.688, -0.525, 0.4999], [-0.162, -0.850, -0.499],
                      [-0.688, -0.525, -0.499], [0.2628, -0.525, -0.809],
                      [0.9518, 0.0000, -0.309], [0.9510, 0.0000, 0.3090],
                      [0.5876, 0.0000, 0.8090], [0.0000, 0.0000, 1.0000],
                      [-0.588, 0.0000, 0.8090], [-0.951, 0.0000, 0.3090],
                      [-0.955, 0.0000, -0.309], [-0.587, 0.0000, -0.809],
                      [0.0000, 0.0000, -1.000], [0.5877, 0.0000, -0.809],
                      [0.6889, 0.5257, 0.4999], [-0.262, 0.5257, 0.8090],
                      [-0.854, 0.5257, 0.0000], [-0.262, 0.5257, -0.809],
                      [0.6889, 0.5257, -0.499], [0.5257, 0.8506, 0.0000],
                      [0.1626, 0.8506, 0.4999], [-0.425, 0.8506, 0.3090],
                      [-0.422, 0.8506, -0.309], [0.1624, 0.8506, -0.499]])
        f = np.array([[15, 3, 13], [13, 14, 15], [2, 15, 14], [13, 1, 14], [17, 2, 14], [14, 16, 17],
                      [6, 17, 16], [14, 1, 16], [19, 4, 18], [18, 13, 19], [3, 19, 13], [18, 1, 13],
                      [21, 5, 20], [20, 18, 21], [4, 21, 18], [20, 1, 18], [22, 6, 16], [16, 20, 22],
                      [5, 22, 20], [16, 1, 20], [24, 2, 17], [17, 23, 24], [11, 24, 23], [23, 17, 6],
                      [26, 3, 15], [15, 25, 26], [7, 26, 25], [25, 15, 2], [28, 4, 19], [19, 27, 28],
                      [8, 28, 27], [27, 19, 3], [30, 5, 21], [21, 29, 30], [9, 30, 29], [29, 21, 4],
                      [32, 6, 22], [22, 31, 32], [10, 32, 31], [31, 22, 5], [33, 7, 25], [25, 24, 33],
                      [11, 33, 24], [24, 25, 2], [34, 8, 27], [27, 26, 34], [7, 34, 26], [26, 27, 3],
                      [35, 9, 29], [29, 28, 35], [8, 35, 28], [28, 29, 4], [36, 10, 31], [31, 30, 36],
                      [9, 36, 30], [30, 31, 5], [37, 11, 23], [23, 32, 37], [10, 37, 32], [32, 23, 6],
                      [39, 7, 33], [33, 38, 39], [12, 39, 38], [38, 33, 11], [40, 8, 34], [34, 39, 40],
                      [12, 40, 39], [39, 34, 7], [41, 9, 35], [35, 40, 41], [12, 41, 40], [40, 35, 8],
                      [42, 10, 36], [36, 41, 42], [12, 42, 41], [41, 36, 9], [38, 11, 37], [37, 42, 38],
                      [12, 38, 42], [42, 37, 10]]) - 1

        # Broadcast vertices to all sphere centers
        vertices = (self.radius * v[np.newaxis, :, :] + self.points[:, np.newaxis, :]).reshape(-1, 3)

        # Adjust face indices for multiple spheres
        sphere_count = self.points.shape[0]
        faces = (f[np.newaxis, :, :] + (np.arange(sphere_count) * v.shape[0])[:, np.newaxis, np.newaxis]).reshape(-1, 3)

        return Mesh(vertices=vertices, faces=faces, vc=self.vc)

# def Arrow(radius=0.02, length=0.27, head_radius=0.035, head_length=0.1, segments=16):
def Arrow(radius=0.03, length=0.5, head_radius=0.055, head_length=0.2, segments=16):

    """
    Create an arrow mesh with cylindrical body and conical head pointing in +z direction.
    Returns vertices and faces arrays suitable for trimesh format.

    Parameters:
    -----------
    radius : float
        Radius of the cylinder body
    length : float
        Total length of the arrow (including head)
    head_radius : float
        Radius of the cone base
    head_length : float
        Length of the cone head
    segments : int
        Number of segments around the circumference

    Returns:
    --------
    vertices : np.array
        Array of vertex positions shape (N, 3)
    faces : np.array
        Array of face indices shape (M, 3)
    """
    # Calculate body length
    body_length = length - head_length

    # Generate cylinder vertices
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    # Bottom circle
    bottom_circle = np.column_stack((
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta)
    ))
    # Top circle
    top_circle = np.column_stack((
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full_like(theta, body_length)
    ))

    # Generate cone vertices
    cone_base = np.column_stack((
        head_radius * np.cos(theta),
        head_radius * np.sin(theta),
        np.full_like(theta, body_length)
    ))
    # Cone tip
    cone_tip = np.array([[0, 0, length]])

    # Combine all vertices
    vertices = np.vstack((bottom_circle, top_circle, cone_base, cone_tip))

    # Generate cylinder faces
    cylinder_faces = []
    for i in range(segments):
        # Bottom face
        cylinder_faces.append([i, (i + 1) % segments, segments + i])
        cylinder_faces.append([segments + i, (i + 1) % segments, segments + (i + 1) % segments])

        # Wall faces
        bottom_idx = i
        top_idx = segments + i
        next_bottom_idx = (i + 1) % segments
        next_top_idx = segments + ((i + 1) % segments)
        cylinder_faces.extend([
            [bottom_idx, next_bottom_idx, top_idx],
            [next_bottom_idx, next_top_idx, top_idx]
        ])

    # Generate cone faces
    cone_faces = []
    cone_base_start_idx = segments * 2
    cone_tip_idx = len(vertices) - 1
    for i in range(segments):
        # Side faces
        cone_faces.append([
            cone_base_start_idx + i,
            cone_base_start_idx + ((i + 1) % segments),
            cone_tip_idx
        ])

    # Combine all faces
    faces = np.array(cylinder_faces + cone_faces)

    return Mesh(vertices=vertices, faces=faces)

def Floor(center=(0, 0, 0), scale=(1, 1), vc=(1, 1, 1), y_up=False):
    """
    Create a floor mesh with a specified center and scale.

    Parameters:
    center (tuple): The (x, y, z) coordinates of the floor center.
    scale (tuple): The (width, depth) scale factors of the floor.

    Returns:
    trimesh.Trimesh: The created floor mesh.
    """
    cx, cy, cz = center
    sx, sy = scale

    # Define the vertices of the rectangle (floor)
    vertices = np.array([
        [cx - sx / 2, cy - sy / 2, cz],
        [cx + sx / 2, cy - sy / 2, cz],
        [cx + sx / 2, cy + sy / 2, cz],
        [cx - sx / 2, cy + sy / 2, cz]
    ])
    if y_up:
        vertices = vertices[:, [0, 2, 1]]

    # Define the faces of the rectangle (2 triangles)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # Create the mesh
    floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vc=vc)

    return floor_mesh


def FloorMesh(offset=0, scale=1, height_max=0, height_min=0, y_up=False):
    """
    Create a floor mesh with a specified center, scale, and height range.

    Parameters:
    center (tuple): The (x, y, z) coordinates of the floor center.
    scale (tuple): The (width, depth) scale factors of the floor.
    height_max (float): The maximum height of the floor.
    height_min (float): The minimum height of the floor.
    y_up (bool): If True, use Y-up coordinate system.

    Returns:
    trimesh.Trimesh: The created floor mesh.
    """
    cx, cy, cz = 0, 0, 0

    sx = sy = scale

    # Calculate the average height for the center
    avg_height = (height_max + height_min) / 2
    cz += avg_height

    # Define the vertices of the rectangle (floor)
    vertices = np.array([
        [cx - sx / 2, cy - sy / 2, height_min],
        [cx + sx / 2, cy - sy / 2, height_min],
        [cx + sx / 2, cy + sy / 2, height_min],
        [cx - sx / 2, cy + sy / 2, height_min],
        [cx - sx / 2, cy - sy / 2, height_max],
        [cx + sx / 2, cy - sy / 2, height_max],
        [cx + sx / 2, cy + sy / 2, height_max],
        [cx - sx / 2, cy + sy / 2, height_max]
    ])

    if y_up:
        vertices = vertices[:, [0, 2, 1]]

    # Define the faces of the cube
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom face
        [4, 5, 6], [4, 6, 7],  # top face
        [0, 4, 7], [0, 7, 3],  # left face
        [1, 5, 6], [1, 6, 2],  # right face
        [0, 1, 5], [0, 5, 4],  # front face
        [3, 2, 6], [3, 6, 7]  # back face
    ])

    # Create the mesh
    floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return floor_mesh


colors = {
    'pink': [1.00, 0.75, 0.80],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'yellow': [1.0, 1.0, 0.0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [0.0, 0.0, 1.0],
    'lightblue': [0.678, 0.84, 0.90],
    'white': [1.0, 1.0, 1.0],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0.0, 0.0, 0.0],
    'cyan': [0.0, 1.0, 1.0],
    'magenta': [1.0, 0.0, 1.0],
    'lime': [0.0, 1.0, 0.0],
    'indigo': [0.29, 0.0, 0.51],
    'violet': [0.56, 0.0, 1.0],
    'gold': [1.0, 0.84, 0.0],
    'silver': [0.75, 0.75, 0.75],
    'beige': [0.96, 0.96, 0.86],
    'maroon': [0.5, 0.0, 0.0],
    'navy': [0.0, 0.0, 0.5],
    'teal': [0.0, 0.5, 0.5],
    'olive': [0.5, 0.5, 0.0],
    'coral': [1.0, 0.5, 0.31],
    'salmon': [0.98, 0.5, 0.45],
    'khaki': [0.94, 0.9, 0.55],
    'lavender': [0.9, 0.9, 0.98],
    'turquoise': [0.25, 0.88, 0.82],
    'peach': [1.0, 0.87, 0.68],
    'plum': [0.87, 0.63, 0.87],
}


def Axis(scale=0.5):
    vc = [colors['red'], colors['green'], colors['blue']]
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
    e = np.array([[0, 1], [0, 2], [0, 3]])
    return [Lines(vertices=v, indices=e)]
