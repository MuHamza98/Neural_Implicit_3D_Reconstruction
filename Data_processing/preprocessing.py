'''

Functions to do:

1. Progressive Training optional (Inspired from IM-NET) 
2. Convert meshes to watertight meshes using TDF


'''
import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import polyscope as ps
import igl




########################################                     AUXILLARY FUNCTIONS              ########################################



def create_grid_points_from_bounds(data_params):

	'''
	Input:
	minimum: float; 
	maximum: float;
	res: int;

	These determine the size and resolution of the grid which will be used to store occupancy values.

	Output: res x 3 

	'''

	maximum, minimun, res = data_params['bb_max'], data_params['bb_min'], data_params['input_res']
	x = np.linspace(minimun, maximum, res)
	X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
	X = X.reshape((np.prod(X.shape),))
	Y = Y.reshape((np.prod(Y.shape),))
	Z = Z.reshape((np.prod(Z.shape),))

	points_list = np.column_stack((X, Y, Z))
	return points_list


def get_mesh_names(dirname):
	'''
	Input:
	dirname: str; the dir where all meshes can be found

	Get a list of mesh names; currently gets all files with extension : .obj, .off
	'''
	files = []
	extensions = ['.obj', '.off', '.ply']
	for root, _, filenames in os.walk(dirname):
		for ext in extensions:
			for file in filenames:
				if file.endswith(ext):
					files.append(os.path.join(root,file))
	return files


def as_mesh(scene_or_mesh):
	"""
	Convert a possible scene to a mesh.

	If conversion occurs, the returned mesh has only vertex and face data.
	Suggested by https://github.com/mikedh/trimesh/issues/507
	"""
	if isinstance(scene_or_mesh, trimesh.Scene):
		if len(scene_or_mesh.geometry) == 0:
			mesh = None  # empty scene
		else:
			# we lose texture information here
			mesh = trimesh.util.concatenate(
				tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
					for g in scene_or_mesh.geometry.values()))
	else:
		assert(isinstance(scene_or_mesh, trimesh.Trimesh))
		mesh = scene_or_mesh
	return mesh



def read_mesh(mesh_names):
	'''
	Input:
	mesh_names: [mesh_file, ...., mesh_file]; 

	Output:
	Returns a list of Trimesh mesh objects.

	'''
	meshes = []
	for m_name in mesh_names:
		temp = as_mesh(trimesh.load(m_name))
		meshes.append(temp)
	return meshes


def dist_between_point_clouds(cloud_1, cloud_2):

	cloud_1_kd_tree = KDTree(cloud_1)
	cl1_cl2_distances, _ = cloud_1_kd_tree.query(cloud_2)
	return cl1_cl2_distances


def normalizing_meshes(list_of_meshes, mesh_names, out_dir, input_dir):
	'''
	Input:
	list_of_meshes: list; contains trimesh objects
	mesh_names: list; contains strings of meshes names + locations
	out_dir: the directory we want to save our meshes in 
	input_dir: str; we save the files in the same format as we found it but instead of input_dir we use output_dir; this replaces the input_dir in mesh name with the output 

	We use the method as presented in NDF; 

	Output:
	List of meshes where each mesh has been normalized -> mesh is divided by total size; mesh is translated by the center

	'''


	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	

	for i in range(len(list_of_meshes)):
		mesh = list_of_meshes[i]
		try:
			total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
			centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

			mesh.apply_translation(-centers)
			mesh.apply_scale(1 / total_size)
			mesh_name = mesh_names[i]
			list_of_meshes[i] = mesh
		except:
			continue


		mesh_name = mesh_name.replace(input_dir, out_dir)
		
		dir_ = os.path.dirname(mesh_name)
		
		basename = os.path.basename(mesh_name)
		basename, ext = os.path.splitext(basename)
		out_file = os.path.join(dir_, basename + '_normalized' + ext)
		if not os.path.exists(dir_):
			os.makedirs(dir_)

		try:
			mesh.export(out_file)
		except:
			out_file = out_file.replace(ext, '.ply')
			mesh.export(out_file)

	return list_of_meshes


def load_voxels(voxel_names):
	'''
	Use trimesh 
	we may even be able to just use normalize as is 
	'''

	list_of_voxels = []

	for voxel in voxel_names:
		vox = trimesh.load(voxel)
		vox = vox.fill()
		list_of_voxels.append(vox)
	return list_of_voxels


def find_voxels(dirname):

	files = []
	extensions = ['surface.binvox']
	for root, _, filenames in os.walk(dirname):
		for ext in extensions:
			for file in filenames:
				if file.endswith(ext):
					files.append(os.path.join(root,file))
	return files



def downsize_voxel(voxel, size):

	voxel = voxel.matrix.astype(int)

	dims = voxel.shape[0]
	if dims == size:
		return voxel
	else:
		smaller_voxel = np.zeros([size,size,size],np.uint8)
		multiplier = int(dims/size)
		for i in range(size):
			for j in range(size):
				for k in range(size):
					smaller_voxel[i,j,k] = np.max(voxel[ i*multiplier:(i+1)*multiplier, j*multiplier:(j+1)*multiplier,  k*multiplier:(k+1)*multiplier ])
	
	return smaller_voxel


########################################                     MAIN FUNCTIONS              ########################################


def sample_points(list_of_meshes, mesh_names, data_params, exp_name, input_type = "mesh"):
	'''
	Input:
	list_of_meshes: list; contains trimesh objects
	mesh_names: list; contains names of meshes; these are then used to save details!
	num_points: int; total num of points per mesh to be generated;
	noise: np array; how much gaussian std dev should be added to the points; 
	ratio: list; how many points corresponding to each noise dist
	distance_function: boolean; Indicates whether we should save ground truth distance functions
	input_type: whether input is mesh or point cloud

	Output:
	Returns
	 
	Saves
	dict:   points: list_of_generated_points
			dist_func: calculated distance function for each point

	'''
	
	num_points = data_params['num_points']
	noise =  np.array(data_params['noise'])
	ratio = np.array(data_params['ratio'])
	distance_function = data_params['df']

	out_dir = os.path.join(data_params['out_dir'], exp_name) + '/GT/'
	input_dir = data_params['dirname']

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	points_per_noise = num_points * ratio
	points_per_noise = np.round(points_per_noise).astype(int)
	for x in range(len(list_of_meshes)):

		mesh_points, mesh_df = [], []
		mesh = list_of_meshes[x]

		for i in range(len(noise)):
			
			if input_type == "mesh" :
				points = mesh.sample(points_per_noise[i])
			elif input_type == "point_cloud" :
				points = get_points_from_point_cloud(mesh, points_per_noise[i])
			df = np.array([])
			if distance_function:
				df = np.zeros(points_per_noise[i])
			sigma = noise[i]

			if sigma != 0:
				if input_type == "mesh" :
					points = points + sigma * np.random.randn(points_per_noise[i], 3)
					if distance_function:
						df = np.abs(igl.signed_distance(points, mesh.vertices, mesh.faces)[0])
				elif input_type == "point_cloud":
					new_points = points + sigma * np.random.randn(points_per_noise[i], 3)
					if distance_function:
						df = dist_between_point_clouds(points, new_points)
					points = new_points
					
			mesh_points.append(points)
			mesh_df.append(df)
			


		mesh_name = mesh_names[x]
		mesh_name = mesh_name.replace(input_dir, out_dir)
		dir_ = os.path.dirname(mesh_name)
		basename = os.path.basename(mesh_name)
		basename, ext = os.path.splitext(basename)
		out_file = os.path.join(dir_, basename + '_points.npz')
		if not os.path.exists(dir_):
			os.makedirs(dir_)

		np.savez(out_file, points=np.array(mesh_points, dtype = object), df = np.array(mesh_df, dtype = object), ratio = ratio, noise = noise)
		
		#list_of_mesh_points.append(np.array(mesh_points, dtype = object))




		# CURRENTLY WE DO NOT HAVE GRID_COORDS AS BUILT IN THE NDF. WE COULD FIND NO REASON TO DO THIS AND THEREFORE WILL CONTINUE WITH THIS UNLESS
		# WE GET POOR RESULTS OR SOME OTHER EXPLANATION IS FORTHCOMING



################## Create Voxels from Mesh ############################
def voxels_from_mesh(list_of_meshes, mesh_names, kdtree, data_params, exp_name):
	'''
	Input:
	list_of_meshes: list; contains either the trimesh.trimesh objects or sampled point_clouds ( see num_points)
	mesh_names: list; contains names of filename of each mesh; 
	kdtree:  the kd tree that has partitioned the grid which we want to label with occupancy values
	num_points: int; num of point to sample from the mesh; if 0 this implies  list_of_meshes are in fact point clouds


	Output:
	Creates a voxel grid of specified dimensions and resolution; then using the mesh the occupancy values of the grid are determined
	'''
	

	bb_max, bb_min, input_res, num_points = data_params['bb_max'], data_params['bb_min'], data_params['input_res'], data_params['num_points_voxel']
	out_dir = os.path.join(data_params['out_dir'], exp_name) + '/GT/'

	input_dir = data_params['dirname']

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)


	for i in range(len(list_of_meshes)):

		if num_points == 0:
			point_cloud = np.array(list_of_meshes[i].vertices)

		else:
			point_cloud = list_of_meshes[i].sample(num_points)
		
		occupancies = np.zeros(input_res **3, dtype=np.int8) # we do not have grid_points; this is input_res**3

		_, idx = kdtree.query(point_cloud)
		occupancies[idx] = 1


		compressed_occupancies = np.packbits(occupancies)

		mesh_name = mesh_names[i]
		mesh_name = mesh_name.replace(input_dir, out_dir)
		dir_ = os.path.dirname(mesh_name)
		basename = os.path.basename(mesh_name)
		basename, ext = os.path.splitext(basename)
		out_file = os.path.join(dir_, basename + '_voxels.npz')
		if not os.path.exists(dir_):
			os.makedirs(dir_)
		np.savez(out_file, compressed_occupancies = compressed_occupancies, bb_min = bb_min, bb_max = bb_max, input_res = input_res)



################## Create Point Value pairs from Voxels ############################
def surface_points_from_voxel(list_of_voxels, voxel_names, params, exp_name):


	'''
	list_of_voxels: list of voxels (3D matrices)
	voxel_names: list of names for voxels
	params:
	num_points : num of point value pairs to extract
	split : ratio of surface points to randomly chosen points
	window_size : how large a patch to consider when deciding surface points
	progressive training: the voxel sizes to downsize to for progressive training later on


	'''
	num_points =  params['num_points']
	split =  params['split']
	window_size = params['window_size']
	input_dir = params['dirname']
	output_dir = os.path.join(params['out_dir'], exp_name) + '/GT/'
	progressive_training = params['prog_training']
	surface_points = int(split[0] * num_points)

	for size in progressive_training:

		for i in range(len(list_of_voxels)):

			voxel_name = voxel_names[i]
			voxel_name = voxel_name.replace(input_dir, output_dir)
			dir_ = os.path.dirname(voxel_name)
			basename = os.path.basename(voxel_name)
			basename, _ = os.path.splitext(basename)

			out_file = os.path.join(dir_, basename + '_' + str(size) + '_voxel_point_val.npz')


			if os.path.exists(out_file):
				continue


			voxel = list_of_voxels[i]
			try:
				voxel = downsize_voxel(voxel, size)
			except:
				continue

			dims = voxel.shape
			sample_points, sample_vals = [], []

			############ GET POINTS NEAR SURFACE ############
			while len(sample_points) < surface_points:

				indices = np.random.randint(window_size, np.array(dims) - window_size, (surface_points, 3)) 
				for idx in indices:
					i,j,k = idx[0], idx[1], idx[2]

					voxel_window = voxel[i -window_size: i+window_size, j - window_size: j + window_size, k - window_size: k + window_size] # extract path from voxels 
					surface = np.max(voxel_window) != np.min(voxel_window) # if the window contains inside shape points as well as outside shape points the idx is on the surface
					if surface:
						sample_points.append(idx)
						sample_vals.append(voxel[i,j,k])

					if len(sample_points) == surface_points:
						break
			
			############ GET POINTS RANDOMLY  ############
			indices = np.random.randint(0, dims, (num_points - surface_points, 3)) # 3D points hence 3   
			for idx in indices:
				i,j,k = idx[0], idx[1], idx[2]
				sample_points.append(idx)
				sample_vals.append(voxel[i,j,k])



			if not os.path.exists(dir_):
				os.makedirs(dir_)
			
			np.savez(out_file, sample_points = sample_points, sample_vals = sample_vals)

'''

		mesh_name = mesh_name.replace(input_dir, out_dir)
		dir_ = os.path.dirname(mesh_name)
		basename = os.path.basename(mesh_name)
		basename, ext = os.path.splitext(basename)
		out_file = os.path.join(dir_, basename + '_normalized' + ext)
		if not os.path.exists(dir_):
			os.makedirs(dir_)

		try:
			mesh.export(out_file)
		except:
			out_file = out_file.replace(ext, '.ply')
			mesh.export(out_file)

'''

def get_PC_names(dirname):
	'''
	Input:
	dirname: str; the dir where all point clouds can be found

	Get a list of mesh names; currently gets all files with extension : .txt
	'''
	files = []
	extensions = ['.txt']
	for root, _, filenames in os.walk(dirname):
		for ext in extensions:
			for file in filenames:
				if file.endswith(ext):
					files.append(os.path.join(root,file))
	return files


def read_PC(pc_names):
	'''
	Read each file using numpy; 
	Expected format:
	X_i, Y_i, Z_i
	X_i+1, Y_i+1, Z_i+1

	Only first 3 cols considered;
	Looks for ',' seperating values

	'''
	pcs = []
	for pc_name in pc_names:
		pc = np.loadtxt(pc_name, usecols = (0,1,2))
		pc = trimesh.PointCloud(pc)
		pcs.append(pc)
	
	return pcs



def get_points_from_point_cloud(pc, num_points):

	points = pc.vertices
	indices = np.random.randint(0, points.shape[0], num_points)  # get indices of random points 
	points = points[indices]
	return points

