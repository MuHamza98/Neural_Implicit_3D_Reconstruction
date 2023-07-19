
import torch
import os
from Data_processing.Dataset import *
from torch.utils.data import DataLoader
from Generator.generator import generate


def generate_outputs(gen_params, exp_name):

    '''
        'model_arc' : training_params['model_arc'], #'Model_Architectures.Model',
        'class' : training_params['class'], #'NDF',
        'model_path'    : training_params['model_path'],# '../Models/NDF_modified/experiments/NDF_modified_exp_6/checkpoints/checkpoint_0h:0m:0s_0.tar', # path to model
        'dir'           : training_params['dir'], #'../data_subset/exp4',
        'modes'         : ['train', 'val', 'test'], # which split should be generated; 
        'voxels': True, # Is this input required?
        'point_cloud': False, # Is this input required?
        'dataset_ratio': [1.0, 0.0, 0.0],
        'num_points': 900000

    '''




# ==================================================================================================================== #
#                                               LOAD SPECIFIED MODEL                                                   #
# ==================================================================================================================== #


    model_path = gen_params['model_path']
    possible_models = [i for i in os.listdir(model_path)if i.endswith('.ckpt')]
    possible_models.sort()
    model_path = os.path.join(model_path, possible_models[-1])
    print('model_path: ', model_path)



    mod = __import__(gen_params['model_arc'], fromlist = [gen_params['class']])
    nn = getattr(mod, gen_params['class'])
    model = nn()


    checkpoint = torch.load(model_path, map_location='cpu')

    new_dict = {}
    for old_key in checkpoint['state_dict']:
        new_key = old_key.replace('model.', '')
        new_dict[old_key] = new_key
    
    model_state = dict((new_dict[key], value) for (key, value) in checkpoint['state_dict'].items())

    model.load_state_dict(model_state)
    model.to(torch.device('cuda:1'))


# ==================================================================================================================== #
#                                               SPLIT THE DATASET INTO SPECIFIED RATIOS                                #
# ==================================================================================================================== #


    split_file_name =  gen_params['dir'] + "/split_" + exp_name  + ".npz"

    print('split_file_name: ', split_file_name)

    if not os.path.isfile(split_file_name):
        print('No existing split found! Do you wish to continue?')
        if input('y/n') == 'n':
            exit()
        split_dataset(gen_params, exp_name)
        print('Split Dataset!')




# ==================================================================================================================== #
#                                               GENERATE OUTPUT                                                        #
# ==================================================================================================================== #
    for mode in gen_params['modes']:
        temp_params = gen_params
        temp_params['mode'] = mode
        gen_dataset = generic_dataset(temp_params, exp_name)
        generate(model, DataLoader(gen_dataset), gen_params)


