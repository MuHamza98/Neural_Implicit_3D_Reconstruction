import matplotlib.pyplot as plt
import polyscope as ps
import numpy as np
import trimesh
import os 

def visualize_meshes(list_of_meshes):
	'''
	list_of_meshes: list of trimesh.trimesh objects; these are cast as a scene and then show

	'''
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	for mesh in list_of_meshes:
		ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
		plt.show()



def visualize_point_cloud(list_of_point_clouds):
	'''
	list_of_point_clouds: list of point_clouds (see below)
	point_cloud: list of numpy array; each numpy array is (n, 3)
	'''
	for points in list_of_point_clouds:
		ps.init()
		for i, points_list in enumerate(points):
			ps.register_point_cloud("my points {}".format(i), points_list)
		
		ps.show()

def visualize_voxels(list_of_voxel_grid):
	'''

	'''
	for voxels in list_of_voxel_grid:
		ax = plt.figure().add_subplot(projection='3d')
		ax.voxels(voxels,  edgecolor='k')
		plt.show()
