import os 
from Data_processing.preprocessing import *


def preprocess_data(params, exp_name):

    '''

    params: dictionary containing the following fields:

    'input'             : 'mesh'                                                # Type of input (mesh, voxels, point clouds)
    'steps'             : ['normalize', 'point_cloud' , 'occupancy_voxels']     # What to do with the input
    'dirname'           : '../data_subset/Subset_copy/Shapenet'                 # Where the data is placed
    'out_dir'           : '../data_subset/'                                     # Where to save the preprocessed Data
    'bb_max'            : 0.5                                                   # In case of voxels, the max size of the bounding box               
    'bb_min'            : -0.5                                                  # In case of voxels, the max size of the bounding box
    'input_res'         : 256,                                                  # Resolution of voxel
    'num_points'        : 100000                                                # Size of point clouds
    'num_points_voxel'  : 10000,                                                # Number of points to query voxel
    'noise'             : [0.08, 0.02, 0.003],                                  # Gaussian noise to add to point cloud
    'ratio'             : [0.01, 0.49, 0.5],                                    # Ratio of points correspoinding to noise
    'df'                : True,                                                 # Whether to use a distance function (In contrast to 0,1) 
    '''


# ==================================================================================================================== #
#                                               CHECK AND CREATE PATHS                                                 #
# ==================================================================================================================== #

    dirname =  params['dirname']
    out_dir = os.path.join(params['out_dir'], exp_name) + '/GT/'
    print(out_dir)


    if os.path.exists(out_dir):
        ans = ''
        while ans != 'y' and ans != 'n':
            ans = input('Dir already exists! Are you sure you want to proceed? [y/n]')
        if ans == 'n':
            return
    else:
        os.makedirs(out_dir)





    # ==================================================================================================================== #
    #                                               MESHES                                                                 #
    # ==================================================================================================================== #
    if params['input'] == 'mesh':

        # ==================================================================================================================== #
        #                                               LOAD MESHES                                                            #
        # ==================================================================================================================== #
        try:
            print('\t\t*********** Finding Meshes')
            mesh_names = get_mesh_names(dirname)
            print('\t\t*********** Found {} Meshes'.format(len(mesh_names)))
        except Exception as e:
            print('Problem with finding mesh files! ', e)
            exit()


        batch_size = 100
        for i in range(0, len(mesh_names), batch_size):

            try:
                print('\t\t*********** Loading Meshes')
                list_of_meshes = read_mesh(mesh_names[i:i + batch_size]) 
                print('\t\t*********** Loaded Meshes')
            except Exception as e:
                print('Could not Load mesh files!', e)
                exit()             
            

            # ==================================================================================================================== #
            #                                               NORMALIZE MESHES                                                       #
            # ==================================================================================================================== #
            if 'normalise' in params['steps']:
                try:
                    print('\t\t*********** Normalizing Meshes')
                    list_of_meshes = normalizing_meshes(list_of_meshes, mesh_names[i:i + batch_size], out_dir, dirname)
                    print('\t\t*********** Saved {} Meshes'.format(len(list_of_meshes)))
                except Exception as e:
                    print('Issue with normalzing meshes!')
                    print(e)               
                    exit()
                

            # ==================================================================================================================== #
            #                                               VOXELIZE MESHES                                                        #
            # ==================================================================================================================== #
            if 'occupancy_voxels' in params['steps']:
                print('\t\t*********** Voxelizing Meshes!')

                grid = create_grid_points_from_bounds(params)
                kdtree = KDTree(grid)
                try:
                    
                    voxels = voxels_from_mesh(list_of_meshes, mesh_names[i:i + batch_size], kdtree, params, exp_name)
                    print('\t\t*********** Voxelized {} Meshes'.format(len(list_of_meshes)))
                except Exception as e:
                    print('Issue with voxelizing meshes! ', e)
                    exit()

            # ==================================================================================================================== #
            #                                               CREATE POINT CLOUD FROM MESHES                                         #
            # ==================================================================================================================== #
            if 'point_cloud' in params['steps']:
                try:
                    print('\t\t*********** Creating Point Clouds!')
                    points = sample_points(list_of_meshes, mesh_names[i:i + batch_size], params, exp_name)
                    print('\t\t*********** Sampled boundary points of {} Meshes'.format(len(list_of_meshes)))
                except Exception as e:
                    print('Issues with sampling points: ', e)
                    exit()               


    # ==================================================================================================================== #
    #                                               VOXELS                                                                 #
    # ==================================================================================================================== #
    elif params['input'] == 'voxels':

        # ==================================================================================================================== #
        #                                               LOAD VOXELS                                                            #
        # ==================================================================================================================== #
        try:
            print('\t\t*********** Finding Voxels')
            voxel_names = find_voxels(dirname)
            print('\t\t*********** Found {} Voxels'.format(len(voxel_names)))
        except Exception as e:
            print('Problem with finding Voxels files! ', e)
            exit()

        batch_size = 100
        for i in range(0, len(voxel_names), batch_size):

            try:
                print('\t\t*********** Loading Voxels')
                list_of_voxels = load_voxels(voxel_names[i:i + batch_size]) 
                print('\t\t*********** Loaded Voxels')
            except Exception as e:
                print('Could not Load Voxels!', e)
                exit()  


            # ==================================================================================================================== #
            #                                               NORMALIZE Voxels                                                       #
            # ==================================================================================================================== #
            if 'normalise' in params['steps']:
                try:
                    print('\t\t*********** Normalizing Voxels')
                    list_of_voxels = normalizing_meshes(list_of_voxels, voxel_names[i:i + batch_size], out_dir, dirname)
                    print('\t\t*********** Saved {} Voxels'.format(len(list_of_voxels)))
                except Exception as e:
                    print('Issue with normalizing Point Clouds!')
                    print(e)               
                    exit()

            # ==================================================================================================================== #
            #                                               Point Value pairs                                                      #
            # ==================================================================================================================== #
            if 'point_value' in params['steps']:
                try:
                    print('\t\t*********** Creating Occupancy Pairs!')
                    surface_points_from_voxel(list_of_voxels, voxel_names[i:i + batch_size], params, exp_name)
                    print('\t\t*********** Saved {} Occupancy Pairs'.format(len(list_of_voxels)))
                except Exception as e:
                    print('Issue with creating Occupancy Pairs!')
                    print(e)               
                    exit()


    # ==================================================================================================================== #
    #                                               POINT CLOUDS                                                           #
    # ==================================================================================================================== #
    elif params['input'] == 'point_cloud':


        # ==================================================================================================================== #
        #                                               LOAD POINT CLOUDS                                                      #
        # ==================================================================================================================== #
        try:
            print('\t\t*********** Finding Point Clouds')
            pc_names = get_PC_names(dirname)
            print('\t\t*********** Found {} Point Clouds'.format(len(pc_names)))
        except Exception as e:
            print('Problem with finding Point Clouds files! ', e)
            exit()

        batch_size = 50
        for i in range(0, len(pc_names), batch_size):
            try:
                print('\t\t*********** Loading Point Clouds')
                list_of_pcs = read_PC(pc_names[i:i + batch_size]) 
                print('\t\t*********** Loaded Point Clouds')
            except Exception as e:
                print('Could not Load Point Cloud files!', e)
                exit()  


            # ==================================================================================================================== #
            #                                               NORMALIZE POINT CLOUDS                                                 #
            # ==================================================================================================================== #

            if 'normalise' in params['steps']:
                try:
                    print('\t\t*********** Normalizing Point Clouds')
                    list_of_pcs = normalizing_meshes(list_of_pcs, pc_names[i:i + batch_size], out_dir, dirname)
                    print('\t\t*********** Saved {} Point Clouds'.format(len(list_of_pcs)))
                except Exception as e:
                    print('Issue with normalizing Point Clouds!')
                    print(e)               
                    exit()



            # ==================================================================================================================== #
            #                                               VOXELIZE POINT CLOUDS                                                  #
            # ==================================================================================================================== #

            if 'occupancy_voxels' in params['steps']:
                print('\t\t*********** Voxelizing Point Clouds!')

                grid = create_grid_points_from_bounds(params)
                kdtree = KDTree(grid)
                try:
                    voxels_from_mesh(list_of_pcs, pc_names[i:i + batch_size], kdtree, params, exp_name, )
                    print('\t\t*********** Voxelized {} Point Clouds'.format(len(list_of_pcs)))
                except Exception as e:
                    print('Issue with voxelizing Point Clouds! ', e)
                    exit()


            # ==================================================================================================================== #
            #                                                NOISY POINT CLOUDS                                                    #
            # ==================================================================================================================== #

            if 'point_cloud' in params['steps']:
                try:
                    print('\t\t*********** Creating Point Clouds!')
                    sample_points(list_of_pcs, pc_names[i:i + batch_size], params, exp_name, "point_cloud")
                    print('\t\t*********** Sampled boundary points of {} Meshes'.format(len(list_of_pcs)))
                except Exception as e:
                    print('Issues with sampling points: ', e)
                    exit()    