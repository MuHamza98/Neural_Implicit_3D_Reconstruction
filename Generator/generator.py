import os
import torch
import numpy as np
import trimesh
from torch.nn import functional as F
import gc


# Model takes generater dataset and produces point clouds/ meshes from that
# One functions for now -> encoder decoder

def generate(model, dataloader, gen_params, num_steps = 10):

    #what does this do?
    '''


    Generation process:

    1. Generate a large number of uniformly distributed points
    2. Copmute distance for those points 
    3. Filter out points with large distances
    4. Select required number of points, randomly and reselecting them over again
    5. Add noise to them and restart over again
    6. Select and add the points 
    for each point in the dataloader pass it to the model ok
    I think we have to fix this function for encoder decoder i.e function all networks that have encoder decoder and use this input will follow this!


    Set model gradients to False


    Put samples down in the dataloader loop; for each data size choose a new samples 

    '''

    for param in model.parameters():
        param.requires_grad = False

    sample_num = 150000
    # list_of_point_clouds = []
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = next(model.parameters()).device


    model_num = 0
    for _, data in enumerate(dataloader):


        stored_points = np.zeros((0, 3)) # actual points to be returned

        # CREATE UNFIRMLY DISTRIBUTED POINTS

        samples = torch.rand(1, sample_num, 3).float().to(device) * 3 - 0.5 # points in between 0.5 and -0.5 
        samples.requires_grad = True



        ####  Note we skip already generated voxels

        path = data['path']
        out_path = [i.replace(gen_params['dir'], gen_params['dir'] + '/Generated').replace('/GT', '') for i in path]
        out_path = [i.replace('voxels.npz', 'points.off') for i in out_path]

        if os.path.exists(out_path[0]):
            print('skipped')
            continue
        dirs = [os.path.dirname(i) for i in out_path]

        print('path: ', path)


        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        
        
        
        
        ###### ADD OPTIONS ENCODING? FURTHER PLEASE NOTICE ENCODINGS COULD BE LIST! FOR LARGER BATCH SIZE MIGHT NOT WORK!!!!
        inputs = data['voxel'].to(device)
        encoding = model.encoder(inputs)
        inputs = inputs.detach()

        ###### ACTUAL GENERATION

        i = 0
        while len(stored_points) < gen_params['num_points']:

            for j in range(num_steps):
                df_pred = torch.clamp(model.decoder(samples, *encoding), max = 0.1) # empircally chosen max; another point where we have to add possible flexibility 

                df_pred.sum().backward() # retain_graph = True

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True

            if not i == 0:
                stored_points = np.vstack((stored_points, samples[df_pred < 0.001].detach().cpu().numpy())) # after the first attempt; keep all points tha that are close enough to predicted space; filter_val

            samples = samples[df_pred < 0.03].unsqueeze(0) # keep points whose pred_dist is less than 0.03; add a dimension
            indices = torch.randint(samples.shape[1], (1, sample_num))  # list of indices of shape (1, sample_num); max value is length of filtered samples
            samples = samples[
                [
                    [0, ] * sample_num
                ], 
                indices] # Selecting points    
            samples += (0.1/ 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule; i.e adding noise to selected points 
            samples = samples.detach()
            samples.requires_grad = True
            i += 1

        trimesh.Trimesh(vertices = stored_points, faces = []).export(out_path[0])
        model_num += 1
        print('model: ', out_path[0])
        del stored_points
        del samples
        gc.collect()
        torch.cuda.empty_cache()

