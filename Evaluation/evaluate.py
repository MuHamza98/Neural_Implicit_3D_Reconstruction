
####### PURPOSE -> COMPUTING MORE COMLICATED METRICS FOR EVALUATING QUALITY OF OUTPUT
####### OUTPUTS INCLUDE RIGHT NOW POINT CLOUDS 



from scipy.spatial import cKDTree as KDTree
import numpy as np
import json
import os
import trimesh
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm


def as_mesh(scene_or_mesh):# from https://github.com/mikedh/trimesh/issues/507
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
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


def get_meshes(gt_root, recon_root):
    '''
    gt_root: the root where we can find ground truth  
    recon_root: the root where we can find ground truth reconstructed meshes;

    expected: both recon and gt directories have the same structure and file names

    reads in all meshes from the recon_root and finds corresponding meshes in ground truth

    output: list of ground truth meshes as well recon meshes

    '''

    recon_list, gt_list = [], []
    ext = '.off'
    
    for root, _, files in os.walk(recon_root):
        for file in files:
            if file.endswith(ext):
                recon_file = os.path.join(root,file)
                gt_file = recon_file.replace(recon_root, gt_root).replace('_points.off', '.obj')

                if os.path.exists(gt_file):
                    gt_list.append(gt_file)
                    recon_list.append(recon_file)
                else:
                    print('File Not found! ', gt_file)
                    return '', ''
                break
    return recon_list, gt_list


def compute_dists(recon_points, gt_points, metrics):
    '''

    computes hausdorff, chamfer


    '''
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points)

    metrics_dict = {}

    if 'chamfer' in metrics:
    
        cd_re2gt = np.mean(re2gt_distances**2)
        cd_gt2re = np.mean(gt2re_distances**2)
        chamfer_dist = 0.5*(cd_re2gt + cd_gt2re)
        metrics_dict['chamfer'] = chamfer_dist
    
    if 'hausdorff' in metrics:
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
        metrics_dict['hausdorff'] = hausdorff_distance   
    
    return metrics_dict



def evaluate(params):

    '''
    roots both recon and ground truth
    metrics: list -> [chamfer, hausdorff]

    for each metrics do inter class stats and over all stats
    i.e mean, median std, min and max

    '''
    print(params['ground_truth_dir'])
    recon_list, gt_list = get_meshes(params['ground_truth_dir'], params['recon_dir'])
    metric_dists = {}
    for i in tqdm(range(len(recon_list))):

        recon_points = trimesh.load(recon_list[i]).vertices
        gt_points = as_mesh(trimesh.load(gt_list[i]))


        gt_points = gt_points.sample(len(recon_points))
        recon_points = minmax_scale(recon_points, feature_range = (-.5, .5))
        gt_points = minmax_scale(gt_points, feature_range = (-.5, .5))

        m_dict = compute_dists(recon_points, gt_points, params['metrics'])

        for key in m_dict:
            try:
                metric_dists[key].append(m_dict[key])
            except:
                metric_dists[key] = m_dict[key]
    
    overall_stats = {}
    for m in params['metrics']:
        dists = np.array(metric_dists[m])
        overall_stats[m] = [dists.mean(), dists.median(), dists.std(), max(dists), min(dists)]
        print('Key: ', m)
        print(overall_stats[m])
    return overall_stats
         