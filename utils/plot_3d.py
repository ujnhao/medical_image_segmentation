#!/usr/bin/python3
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    image = image.astype(np.int16)
    p = image.transpose(2, 1, 0)
    #     p = p[:,:,::-1]

    print(p.shape)
    verts, faces, _, x = measure.marching_cubes_lewiner(p, threshold)  # marching_cubes_classic measure.marching_cubes

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig('3D.jpg')
    # plt.show()


