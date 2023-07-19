import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

class lightning_model(pl.LightningModule):

    def __init__(self, model, train_params):
        super().__init__()
        self.model = model
        # self.voxel = train_params['voxel']
        # self.point_cloud = train_params['point_cloud']
        self.L1_loss, self.L2_loss, self.rmse = False, False, False
        self.IoU_metric, self.chamfer_metric = False, False

        # for each step (training, valid, test) have a set of metrics
        # have a set of params that dictate inputs 

        for loss in train_params['loss_function']:
            if loss == 'l1':
                self.L1_loss = True
            if loss == 'l2':
                self.L2_loss = True
        
        for metric in train_params['metrics']:
            if metric == 'rmse':
                self.rmse = True
        
        self.voxel = train_params['voxels']
        self.point_cloud = train_params['point_cloud'] or train_params['occupancy_pairs']

        
        if not (self.L1_loss or  self.L2_loss):
            raise Exception('Loss function not implemented!')

        
    
    def training_step(self, batch):


        if self.voxel:
            voxels = batch['voxel']
        
        if self.point_cloud:
            points = batch['point']
        
        y = batch['df']
        # loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=self.max_dist),torch.clamp(df_gt, max=self.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        
        if self.voxel and self.point_cloud:
            y_hat = self.model(points, voxels)
        elif self.voxel:
            y_hat = self.model(voxels)
        else:
            y_hat = self.model(points)
        
        metric_loss = {}

        loss = 0.0
        if self.L1_loss:
            loss_temp = torch.nn.L1Loss(reduction='none')(torch.clamp(y_hat, max=0.1),torch.clamp(y, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
            loss += loss_temp.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
            # metric_loss['L1_loss: ', loss]
            metric_loss['L1_loss'] = loss
        
        if self.rmse:
            loss_fn = nn.MSELoss(reduction='none')
            rmse_loss = torch.sqrt(loss_fn(torch.clamp(y_hat, max=0.1), torch.clamp(y, max=0.1)))
            rmse_loss = rmse_loss.sum(-1).mean()
            metric_loss['RMSE_loss'] = rmse_loss
            # metric_loss['RMSE_loss: ', rmse_loss]

        
        if self.L2_loss:
            
            loss += F.mse_loss(y_hat, y)
            metric_loss['L2_loss'] = loss


        self.log_dict(metric_loss, on_epoch = True)

        return loss
    
    def validation_step(self, batch, batch_idx):


        if self.voxel:
            voxels = batch['voxel']
        
        if self.point_cloud:
            points = batch['point']
        
        y = batch['df']

        
        if self.voxel and self.point_cloud:
            y_hat = self.model(points, voxels)
        elif self.voxel:
            y_hat = self.model(voxels)
        else:
            y_hat = self.model(points)
        
        metric_loss = {}
        loss = 0.0
        if self.L1_loss:
            loss_temp = torch.nn.L1Loss(reduction='none')(torch.clamp(y_hat, max=0.1),torch.clamp(y, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
            loss += loss_temp.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
            metric_loss['Validation L1_loss'] = loss
        
        if self.rmse:
            loss_fn = nn.MSELoss(reduction='none')
            rmse_loss = torch.sqrt(loss_fn(torch.clamp(y_hat, max=0.1), torch.clamp(y, max=0.1)))
            rmse_loss = rmse_loss.sum(-1).mean()
            metric_loss['Validation RMSE_loss'] = rmse_loss

        if self.L2_loss:
            loss += F.mse_loss(y_hat, y)
            metric_loss['Validation L2_loss'] = loss

        self.log_dict(metric_loss, on_epoch = True)

        return loss
    
    def test_step(self, batch, batch_idx):

        if self.voxel:
            voxels = batch['voxel']
        
        if self.point_cloud:
            points = batch['point']
        
        y = batch['df']
        
        if self.voxel and self.point_cloud:
            y_hat = self.model(points, voxels)
        elif self.voxel:
            y_hat = self.model(voxels)
        else:
            y_hat = self.model(points)
        
        metric_loss = {}
        loss = 0.0
        if self.L1_loss:
            loss_temp = torch.nn.L1Loss(reduction='none')(torch.clamp(y_hat, max=0.1),torch.clamp(y, max=0.1))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
            loss = loss_temp.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
            metric_loss['Test L1_loss: '] = loss

        if self.L2_loss:
            loss += F.mse_loss(y_hat, y)
            metric_loss['Test L2_loss'] = loss
        
        if self.rmse:
            loss_fn = nn.MSELoss(reduction='none')
            rmse_loss = torch.sqrt(loss_fn(torch.clamp(y_hat, max=0.1), torch.clamp(y, max=0.1)))
            rmse_loss = rmse_loss.sum(-1).mean()
            metric_loss['Test RMSE_loss: '] =  rmse_loss


        self.log_dict(metric_loss, on_epoch = True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer

