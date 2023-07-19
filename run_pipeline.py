from preprocessing import preprocess_data
from training import train_model
from generate import generate_outputs
from Evaluation.evaluate import evaluate
import json 
import os

exp_params = {
    'exp_name'      : 'Shapenet_voxels',        # EXP NAME Shapenet_Whole
    'preprocessing' : False,                 # Do we need to preprocess the data?
    'training'      : True,                 # Do we need to train the model?
    'generate'      : False,                 # Do we need the model to generate something
    'evaluate'      : False,                # Do we need to get the metrics for a generated output 

}
preprocessing_params  = {
    'input'             : 'voxels',                                                # Type of input (mesh, voxels, point clouds)
    'steps'             : ['normalise', 'point_value'],                            # What to do with the input [normalise, occupancy_voxels', 'point_cloud', point_value]
    'dirname'           : '../DATASETS/ShapeNetCore.v2/',                         # Where the data is placed
    'out_dir'           : '../DATASETS',                                        # Where to save the preprocessed Data
    'bb_max'            : 0.5,                                                   # In case of voxels, the max size of the bounding box               
    'bb_min'            : -0.5,                                                  # In case of voxels, the max size of the bounding box
    'input_res'         : 256,                                                   # Resolution of voxel
    'num_points'        : 50000,                                                # Size of point clouds
    'num_points_voxel'  : 10000,                                                      # Number of points to query voxel
    'noise'             : [0.08, 0.02, 0.003],                                   # Gaussian noise to add to point cloud
    'ratio'             : [0.01, 0.49, 0.5],                                     # Ratio of points correspoinding to noise
    'df'                : True,                                                   # Whether to use a distance function (In contrast to 0,1)
    'prog_training'     : [64,32,16],                                             # If using voxels, set a range of resolutions for progressive training 
    'split'             : [0.65,0.35],                                            # If using voxels, set the range of points to be taken from surface vs randomly chosen points
    'window_size'       : 2,                                                      # If using voxels, how large a patch to look at when considering surface
}

training_params = {
        'batch_size': 8,                                                            # batch size
        'num_workers': 0,                                                           # number of workers
        'dir' :  preprocessing_params['out_dir'] + '/Shapenet_voxels/GT',  # the folder containg the preprocessed dataset; 
        'model_arc' : 'Model_Architectures.Model',                                  # the python file where we have the pytorch model to be used
        'class' : 'IM_Net',                                                            # the name of the model in the file 
        'epochs': 300,                                                              # number of epochs for training
        'loss_function' : ['l2'],                                                   # Loss function to optimize the model on 
        'metrics': ['l1', 'rmse'],                                                  # Metrics added to tensorboard during training
        'voxels': True,                                                             # Does the model take in voxels as input
        'occupancy_pairs': True,                                                    # Does the model take in occupancy value and positions?
        'point_cloud': False,                                                        # Does the model takes in point cloud as input
        'model_path' : 'Saved_Models/Shapenet_voxels/', # + exp_params['exp_name'],                    # where to save the model
        'dataset_ratio': [0.7, 0.2, 0.1],                                           # The ratio of the dataset between : [train, val, test]
        'num_sample_points': 15000,                                                 # Number of points extracted from the point cloud
        'continue_train': False,                                                    # continue training from previous checkpoint
        'shuffle': True,                                                            # Shuffle the dataset
        'classes': True,                                                            # Are there classes in the dataset?
        'devices': [0, 1, 2, 3, 4],
        'res' : '64'
}

generator_params = {

    'model_arc'     : training_params['model_arc'],                     # the python file where we have the pytorch model to be used
    'class'         : training_params['class'],                         # the name of the model in the file
    'model_path'    : training_params['model_path'],                    # where to load the model from
    'dir'           : training_params['dir'],                           # the folder containg the preprocessed dataset used to generate the output
    'modes'         : ['train', 'val', 'test'],                         # which split should be generated; 
    'voxels'        : True,                                             # Does the model take in voxels as input
    'point_cloud'   : False,                                            # Does the model takes in point cloud as input
    'dataset_ratio' : training_params['dataset_ratio'],                 # The ratio of the dataset between : [train, val, test]
    'num_points'    : 100000,                                           # Num of points to generate for point cloud

}

evaluation_params = {
    'ground_truth_dir'  : preprocessing_params['dirname'],              # the directory containing the ground truth 
    'recon_dir'         : generator_params['dir'] + '/../Generated',    # the directory containing the reconstructed outputs
    'metrics'           : ['chamfer', 'hausdorff']                      # the metrics of comparison 
}


def main():


    exp_name = exp_params['exp_name'] 

                        ################################################          PREPROCESSING      ################################################
    if exp_params['preprocessing']: 
        preprocess_data(preprocessing_params, exp_name)

                        ################################################           TRAINING      ################################################

    if exp_params['training']:
        train_model(training_params, exp_name)

                        ################################################           Generate      ################################################


    if exp_params['generate']:
        while True:
            try:
                generate_outputs(generator_params, exp_name)
                break
            except:
                pass

                        ################################################           Evaluate      ################################################

    if exp_params['evaluate']: 
        metrics = evaluate(evaluation_params)

################################################ RECORD EXPERIMENT  ################################################

    dir_name = "exp_records/"

    ######## FINDING THE RIGHT DIR #########
    if not os.path.exists(os.path.join(dir_name, exp_name)):
        dir_name = os.path.join(dir_name, exp_name)
        os.mkdir(dir_name)
    else:
        existing_exps = [file for file in os.listdir(dir_name) if exp_name in file]
        existing_exps.sort()
        try:
            exp_num = existing_exps[-1].split('_')[-1] # get the experiment dir that has this in the name (last i.e most recent one)
            exp_num = int(exp_num) + 1
        except:
            exp_num = 1
        exp_name = exp_name + '_' + str(exp_num)
        dir_name = os.path.join(dir_name, str(exp_name))
        os.makedirs(dir_name)

    ######## SAVE EVERYTHING #######

    with open(dir_name + '/Exp_params.json', "w") as f:
        json.dump(exp_params, f, indent = 4)  

    if exp_params['preprocessing']:
        with open( dir_name + '/Preprocess_params.json', "w") as f:
            json.dump(preprocessing_params, f, indent = 4)
            
    if exp_params['training']:
        with open( dir_name + '/Training_params.json', "w") as f:
            json.dump(training_params, f, indent = 4)

    if exp_params['generate']:
        with open( dir_name + '/Generator_params.json', "w") as f:
            json.dump(generator_params, f, indent = 4) 

    if exp_params['evaluate']:
        with open( dir_name + '/Evaluation_params.json', "w") as f:
            json.dump(evaluation_params, f, indent = 4)

        with open(dir_name + '/Evaluation_results.json', "w") as f:
            json.dump(metrics, f, indent = 4)  
                

if __name__ == '__main__':
    main()





