import torch
from Data_processing.Dataset import *
from torch.utils.data import DataLoader
from Model_Architectures.lightning_wrapper import lightning_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os 
'''

Firstly instead of having seperate train dataset params why not incorporate all of them 
Then create dict of params based on the overall result
Also instead of checking the split file every time you create the dataset have a seperate function
'''

def train_model(train_params, exp_name):


    '''

    params: dict containing the following

        'batch_size': 1,                            # batch size
        'num_workers': 0,                           # number of workers
        'dir' :                                     # the folder containg the preprocessed dataset; 
        'model_arc' : 'Model_Architectures.Model',  # the python file where we have the pytorch model to be used
        'class' : 'NDF',                            # the name of the model in the file 
        'epochs': 200,                              # number of epochs for training
        'loss_function' : ['l1'],                   # Loss function to optimize the model on 
        'metrics': ['l1', 'rmse'],                  # Metrics added to tensorboard during training
        'voxels': True,                             # Does the model take in voxels as input
        'point_cloud': True,                        # Whether model takes point cloud as input (Do we need this?)
        'model_path' : 'Saved_Models'                         # where to save the model
        'dataset_ratio': [1.0, 0.0, 0.0],           # The ratio of the dataset between : [train, val, test]
        'num_sample_points': 50000,                 # Number of points extracted from the point cloud
        'continue_train': False,                    # continue training from previous checkpoint
        'shuffle': True,                            # Shuffle the dataset 
        'classes': True                             # Are there classes in the dataset?

    '''


# ==================================================================================================================== #
#                                               CREATE MODEL DIR                                                       #
# ==================================================================================================================== #

    if not os.path.exists(train_params['model_path']):
        os.makedirs(train_params['model_path'])


# ==================================================================================================================== #
#                                               SPLIT THE DATASET INTO SPECIFIED RATIOS                                #
# ==================================================================================================================== #


    split_file_name =  train_params['dir'] + "/split_" + exp_name  + ".npz"
    if not os.path.isfile(split_file_name):
        split_dataset(train_params, exp_name)



# ==================================================================================================================== #
#                                               DATASETS AND DATALOADERS                                               #
# ==================================================================================================================== #


    dataset_params = {
       'voxels'                 : train_params['voxels'],
       'point_cloud'            : train_params['point_cloud'],
       'occupancy_pairs'        : train_params['occupancy_pairs'],
       'num_sample_points'      : train_params['num_sample_points'], 
       'mode'                   : 'train',
       'dir'                    : train_params['dir'],
       'classes'                : train_params['classes'],
       'res'                    : train_params['res']
    }

    train_dataset = generic_dataset(dataset_params, exp_name)
    train_data_loader = DataLoader(
        train_dataset, shuffle = train_params['shuffle'], batch_size = train_params['batch_size'], num_workers = train_params['num_workers']
        )

    ######### VALIDATION DATASET
    if train_params['dataset_ratio'][1] > 0.0:
        dataset_params['mode'] = 'val'
        val_dataset = generic_dataset(dataset_params, exp_name)
        val_data_loader = DataLoader(
        val_dataset, shuffle = False, batch_size = train_params['batch_size'], num_workers = train_params['num_workers']
        )




    ######### TEST DATASET
    if train_params['dataset_ratio'][2] > 0.0:
        dataset_params['mode'] = 'test'
        test_dataset = generic_dataset(dataset_params, exp_name)
        test_data_loader = DataLoader(
        test_dataset, shuffle = False, batch_size = train_params['batch_size'], num_workers = train_params['num_workers']
        )




# ==================================================================================================================== #
#                                               MODEL TRAINING WITH LIGHTNING                                          #
# ==================================================================================================================== #


    mod = __import__(train_params['model_arc'], fromlist = [train_params['class']])
    nn = getattr(mod, train_params['class'])
    model = nn()

    lit_model = lightning_model(model, train_params)

    devices = train_params['devices']

    for loss in train_params['loss_function']:
        if loss == 'l1':
            val_loss = 'Validation L1_loss'
        elif loss == 'l2':
            val_loss = 'Validation L2_loss'
    

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=val_loss,
        mode="min",
        dirpath= train_params['model_path'],
    )

    # train model
    if train_params['continue_train']:
        # find latest model

        models = [i for i in os.listdir(train_params['model_path']) if i.endswith('.ckpt')]
        models.sort()


        model_path = os.path.join(train_params['model_path'], models[-1])

        trainer = pl.Trainer(
            callbacks = [checkpoint_callback], 
            max_epochs=train_params['epochs'], 
            accelerator="auto", 
            devices = devices, 
            # auto_scale_batch_size = True, 
            auto_lr_find = True, 
            resume_from_checkpoint= model_path, 
            default_root_dir = train_params['model_path']
            ) 
    else:
        trainer = pl.Trainer( callbacks = [checkpoint_callback], max_epochs=train_params['epochs'], accelerator="auto", devices = devices, auto_scale_batch_size  = True, auto_lr_find = True, default_root_dir = train_params['model_path'])
    
    if train_params['dataset_ratio'][1] >= 0.0:
        trainer.fit(model=lit_model, train_dataloaders=train_data_loader, val_dataloaders = val_data_loader)
    else:
        trainer.fit(model=lit_model, train_dataloaders=train_data_loader)
    
    if train_params['dataset_ratio'][2] >= 0.0:
        trainer.test(model=lit_model, dataloaders = test_data_loader)