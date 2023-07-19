import os
import numpy as np
import trimesh

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




def split_dataset(dataset_params, exp_name):

    '''
    Function splits the input files according to the dataset ratio

    1. Get all voxel files (if voxels are required)
    2. Get all point cloud files (if point clouds are required)
    3. If both are required then make sure there are the same number and align their order
       (This requires that file names are same and the only difference is replacing voxel with points )
    4. According to the ratio provided the files are assigned to each split


    '''
    
    
    voxel_paths, point_paths  = [], []
    data_dir = dataset_params['dir']
    ratios = dataset_params['dataset_ratio']
    out_file = data_dir + "/split_" + exp_name + ".npz"
    all_train_voxels, all_train_points, all_val_voxels, all_val_points, all_test_voxels, all_test_points = [], [], [], [], [], []


    if dataset_params['classes']:
        classes = os.listdir(data_dir)

    
    if dataset_params['classes']:
        
        for data_class in classes:
            
            voxel_paths, point_paths  = [], []
            for root, _, files in os.walk(data_dir + '/' + data_class):
                for file in files:
                    if dataset_params['voxels'] and  (file.endswith('voxels.npz') or file.endswith('.binvox')):
                        voxel_paths.append(os.path.join(root, file))
                    if dataset_params['point_cloud'] and file.endswith('points.npz'):
                        point_paths.append(os.path.join(root, file))
                    if dataset_params['occupancy_pairs'] and file.endswith(dataset_params['res'] + '_voxel_point_val.npz'):
                        point_paths.append(os.path.join(root, file))
            


            if dataset_params['voxels'] and (dataset_params['point_cloud'] or dataset_params['occupancy_pairs']):
                if len(point_paths) != len(voxel_paths):
                    raise Exception ('Number of voxel files are not equal to the number of point Clouds!')
                else:
                    voxel_paths.sort()
                    point_paths.sort()
            

            dataset_size = len(voxel_paths)
            train_size = int(ratios[0]*dataset_size)
            val_size = int(ratios[1]*dataset_size)
            # test_size = int(ratios[2]*dataset_size)


            train_voxels = voxel_paths[0 : train_size]
            train_points = point_paths[0 : train_size]
            
            val_voxels = voxel_paths[train_size : train_size + val_size]
            val_points = point_paths[train_size : train_size + val_size]

            test_voxels = voxel_paths[train_size + val_size: ]
            test_points = point_paths[train_size + val_size: ]

            all_train_voxels.extend(train_voxels)
            all_train_points.extend(train_points)

            all_val_voxels.extend(val_voxels)
            all_val_points.extend(val_points)

            all_test_voxels.extend(test_voxels)
            all_test_points.extend(test_points)
                        
            
    else:

        for root, _, files in os.walk(data_dir):
            for file in files:
                if dataset_params['voxels'] and (file.endswith('voxels.npz') or file.endswith('.binvox')):
                    voxel_paths.append(os.path.join(root, file))
                if dataset_params['point_cloud'] and file.endswith('points.npz'):
                    point_paths.append(os.path.join(root, file))
                if dataset_params['occupancy_pairs'] and file.endswith(dataset_params['res'] + '_voxel_point_val.npz'):
                    point_paths.append(os.path.join(root, file)) 
 
        
        if dataset_params['voxels'] and (dataset_params['point_cloud'] or dataset_params['occupancy_pairs']):
            if len(point_paths) != len(voxel_paths):
                raise Exception ('Number of voxel files are not equal to the number of point Clouds!')
            else:
                voxel_paths.sort()
                point_paths.sort()
        

        dataset_size = len(voxel_paths)
        train_size = int(ratios[0]*dataset_size)
        val_size = int(ratios[1]*dataset_size)
        test_size = int(ratios[2]*dataset_size)


        all_train_voxels = voxel_paths[0 : train_size]
        all_train_points = point_paths[0 : train_size]
        
        all_val_voxels = voxel_paths[train_size : train_size + val_size]
        all_val_points = point_paths[train_size : train_size + val_size]

        all_test_voxels = voxel_paths[train_size + val_size: ]
        all_test_points = point_paths[train_size + val_size: ]

    np.savez(out_file, 
    train_voxels = all_train_voxels, train_points = all_train_points, 
    val_voxels = all_val_voxels, val_points = all_val_points,
    test_voxels = all_test_voxels, test_points = all_test_points
    )
    


class generic_dataset():

    def __init__(self, dataset_params, exp_name):
        '''
        dataset_params: dict;
        {
            dir: str; dir where all pre processed dataset is placed
            voxels: bool; does the dataset load voxels
            point_cloud: bool; does the dataset load point_cloud
            num_sample_points: int; how many points per object should be provided? (If greater than the total num of points, then points are reused)
            N.B that number of voxel arrays should be equal to point_cloud arrays; provided they are both used

        }


        '''

        # if not (dataset_params['voxels'] or dataset_params['point_cloud']):
        #     raise Exception('Must have either voxels or points_cloud')
        
        self.voxel = dataset_params['voxels']
        self.point_cloud = dataset_params['point_cloud']
        self.occupancy_pair = dataset_params['occupancy_pairs']
        self.voxel_size = 0
        if dataset_params['res']:
            self.voxel_size = int(dataset_params['res'])
        dirname = dataset_params['dir']
        #self.data_dir = dirname

        self.mode = dataset_params['mode']
        voxel_paths = []
        point_paths = []

        split_file_name =  dirname +  "/split_" + exp_name + ".npz"

        if self.voxel:
            voxel_paths = np.load(split_file_name)[self.mode + '_voxels']
        
        if self.point_cloud:
            point_paths = np.load(split_file_name)[self.mode + '_points']
            self.num_sample_points = dataset_params['num_sample_points']

        if self.occupancy_pair:
            point_paths = np.load(split_file_name)[self.mode + '_points']
            self.num_sample_points = dataset_params['num_sample_points']
        

        
        self.data = {
            'voxels': voxel_paths,
            'point_cloud': point_paths
        }
                


    
    def __len__(self):
        if self.voxel:
            return len(self.data['voxels'])
        else:
            return len(self.data['point_cloud'])           
    

    def __getitem__(self, idx):





        to_return = {
            'path' : self.data['voxels'][idx]
        }


        ####### use trimesh to read binvox file and cast to matrix 
        if self.voxel:
            voxel_path = self.data['voxels'][idx]
            try:
                occupancies = np.unpackbits(np.load(voxel_path, allow_pickle = True)['compressed_occupancies'])
                voxel_grid = np.reshape(occupancies, (np.load(voxel_path)['input_res'], )*3)
                
            except:
                voxel_grid = trimesh.load(voxel_path)
                voxel_grid = voxel_grid.fill()
                voxel_grid = downsize_voxel(voxel_grid, self.voxel_size)
                voxel_grid = np.array(voxel_grid,  dtype=np.float32)
                # voxel_grid = np.expand_dims(voxel_grid, axis=-1)
                
                
            to_return['voxel'] = voxel_grid

        ######### check for point value pair instead 
        ######### in this case we can obtain the pair as and pass all of these 
        if self.point_cloud:
            point_path =  self.data['point_cloud'][idx]#os.path.join(self.data_dir + 'points/' , self.data['point_cloud'][idx])
            points_npz = np.load(point_path, allow_pickle = True)
            sample_dist = points_npz['ratio']
            boundary_samples = points_npz['points']
            points_df = points_npz['df']

            num_samples = np.rint(sample_dist * self.num_sample_points).astype(np.uint32) # rounds to nearest int
            points, df = [], []
            for i, num in enumerate(num_samples):
                
                sample_points = np.array(boundary_samples[i], dtype = np.float32)
                sample_df = np.array(points_df[i], dtype = np.float32)
                subsample_indices = np.random.randint(0, len(boundary_samples[i]), num)
                points.extend(sample_points[subsample_indices])
                df.extend(sample_df[subsample_indices])
            
            to_return['point'] = np.array(points,  dtype=np.float32)
            to_return['df'] = np.array(df,  dtype=np.float32)
        
        if self.occupancy_pair:
            point_path =  self.data['point_cloud'][idx]#os.path.join(self.data_dir + 'points/' s, self.data['point_cloud'][idx])
            points_npz = np.load(point_path, allow_pickle = True)
            boundary_samples = points_npz['sample_points']
            boundary_vals = points_npz['sample_vals']

            sample_ind = np.random.randint(0, len(boundary_samples), self.num_sample_points)

            boundary_samples = np.array(boundary_samples, dtype = np.float32)
            boundary_vals = np.array(boundary_vals, dtype = np.float32)
            points = boundary_samples[sample_ind]
            vals =  boundary_vals[sample_ind]
            to_return['point'] = np.array(points,  dtype=np.float32)
            to_return['df'] = np.array(vals,  dtype=np.float32)




        return to_return
            