import numpy as np
import open3d as o3d
from skimage import morphology


class Voxel:
    def __init__(self, model_path, input_type=None, resolution=-1, filled=None):
        self.model_path = model_path
        self.type = input_type
        self.resolution = resolution

        if self.type == 'pc':
            if model_path.split('.') is not '.npy':
                raise Warning('point cloud data must be numpy files!')
            self.pc_data = np.load(model_path)

        elif self.type == 'dense':
            dense = np.load(self.model_path)
            assert(self.resolution == dense.shape[0])
            pc_data = self.occupancy_to_coords(dense)
            self.pc_data = pc_data

        elif self.type == 'mesh':
            if model_path.split('.') is not '.obj':
                raise Warning('model data must be obj files!')
            mesh = o3d.io.read_triangle_mesh(model_path)
            point_cloud = self.sample_mesh(mesh, resolution=self.resolution)
            self.pc_data = point_cloud

        else:
            raise NotImplementedError

        if filled == 'false':
            dense = self.coords_to_occupancy(self.pc_data, self.resolution)
            filled_dense = self.fill_occupancy(dense)
            filled_pc = self.occupancy_to_coords(filled_dense)
            self.pc_data = filled_pc

    def occupancy_to_coords(self, occupancy):
        occupancy = np.floor(np.array(np.nonzero(occupancy)))
        return occupancy.T

    def coords_to_occupancy(self, coords, resolution):
        dense = np.zeros((resolution, resolution, resolution), dtype=np.int)
        np.put(dense, np.ravel_multi_index(
            coords.astype(np.long).T, dense.shape), 1)
        return dense

    def fill_occupancy(self, occupancy):
        assert len(occupancy.shape) == 3
        filled = np.zeros(occupancy.shape)
        labels = morphology.label(occupancy, background=1, connectivity=1)
        outside_label = np.bincount(labels.flatten()).argmax()
        filled[labels != outside_label] = 1
        return filled

    def sample_mesh(self, mesh, resolution, density=50000):
        vertices = np.asarray(mesh.vertices)
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector(
            (vertices - vmin) / (vmax - vmin).max())
        faces = np.array(mesh.triangles).astype(int)
        vertices = np.array(mesh.vertices)
        vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                             vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
        face_areas = np.sqrt(np.sum(vec_cross**2, 1))
        n_samples = (np.sum(face_areas) * density).astype(int)
        n_samples_per_face = np.ceil(density * face_areas).astype(int)
        floor_num = np.sum(n_samples_per_face) - n_samples
        if floor_num > 0:
            indices = np.where(n_samples_per_face > 0)[0]
            floor_indices = np.random.choice(indices, floor_num, replace=True)
            n_samples_per_face[floor_indices] -= 1
        n_samples = np.sum(n_samples_per_face)
        sample_face_idx = np.zeros((n_samples,), dtype=int)
        acc = 0
        for face_idx, _n_sample in enumerate(n_samples_per_face):
            sample_face_idx[acc:acc + _n_sample] = face_idx
            acc += _n_sample

        r = np.random.rand(n_samples, 2)
        A = vertices[faces[sample_face_idx, 0], :]
        B = vertices[faces[sample_face_idx, 1], :]
        C = vertices[faces[sample_face_idx, 2], :]

        P = (1 - np.sqrt(r[:, 0:1])) * A + \
            np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
            np.sqrt(r[:, 0:1]) * r[:, 1:] * C

        coords = np.floor(P*(resolution-1))
        return coords
