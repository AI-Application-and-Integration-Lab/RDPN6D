import pandas as pd
import json
from PIL import Image
import mmcv
import csv
import open3d as o3d
import numpy as np
DATASETS_ROOT = '/media/sda1/r10922190/RDPN-main/datasets/'
TEST_ROOT = DATASETS_ROOT + '/BOP_DATASETS/tless/test_primesense/'


def load_pointcloud():

    models = []
    for i in range(1, 31):

        pcd = o3d.io.read_point_cloud(
            f'{DATASETS_ROOT}/BOP_DATASETS/tless/models/obj_0000{i:02d}.ply')
        # pcd = np.asarray(pcd.points)

        models.append(pcd)
    return models


cad_models = load_pointcloud()
f = open('/media/sda1/r10922190/RDPN-main/datasets/BOP_DATASETS/tless/test_primesense/test_bboxes/yolox_x_640_tless_real_pbr_tless_bop_test.json')
bboxes = json.load(f)
df = pd.read_csv(
    '/media/sda1/r10922190/RDPN-main/output/gdrn/tlessSO/tlessSO.csv')

for scene_im_id in bboxes.keys():
    scene_id = scene_im_id.split('/')[0]
    im_id = scene_im_id.split('/')[1]

    cam_dict = mmcv.load(TEST_ROOT + f"{int(scene_id):06d}/scene_camera.json")
    cols = 640
    rows = 480
    K = np.array(cam_dict[im_id]["cam_K"]).reshape((3, 3))
    dpt_img = np.array(Image.open(
        TEST_ROOT + f'{int(scene_id):06d}/depth/{int(im_id):06d}.png')).astype(np.float32)

    dpt_img = dpt_img[:, :,  np.newaxis] / 10

    ymap = np.array([[j for i in range(cols)]
                     for j in range(rows)]).astype(np.float32)
    xmap = np.array([[i for i in range(cols)]
                    for j in range(rows)]).astype(np.float32)
    xmap = xmap[:, :, np.newaxis]
    ymap = ymap[:, :, np.newaxis]
    pt2 = dpt_img.astype(np.float32)
    cam_cx = K[0][2]
    cam_cy = K[1][2]
    cam_fx = K[0][0]
    cam_fy = K[1][1]
    pt0 = (xmap - cam_cx) * pt2 / cam_fx
    pt1 = (ymap - cam_cy) * pt2 / cam_fy
    depth_xyz = np.concatenate(
        (pt0, pt1, pt2), axis=2)
    for bbox in bboxes[scene_im_id]:
        if bbox['score'] < 0.3:
            continue
        obj_id = bbox['obj_id']
        print(scene_im_id, obj_id)

        sub_dpt_xyz = depth_xyz  # [int(y1):int(y2), int(x1):int(x2)]
        sub_dpt_xyz = sub_dpt_xyz.reshape(-1, 3)
        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(sub_dpt_xyz)
        trans_init = np.zeros((4, 4))
        try:
            R = df[(df["scene_id"] == int(scene_id)) & (
                df["im_id"] == int(im_id)) & (df["obj_id"] == int(obj_id))]['R'].to_numpy()[0]

            R = np.fromstring(R[0:-2], sep=' ', dtype=float).reshape((3, 3))
            t = df[(df["scene_id"] == int(scene_id)) & (
                df["im_id"] == int(im_id)) & (df["obj_id"] == int(obj_id))]['t'].to_numpy()[0]
            t = np.fromstring(t[0:-2], sep=' ', dtype=float)
        except:
            continue

        trans_init[0:3, 0:3] = R
        trans_init[0:3, 3] = t.T
        trans_init[3, 3] = 1
        tmp = o3d.geometry.PointCloud()
        # np.dot(cad_models[int(bbox['obj_id']) - 1].points, R.T) + t
        # cad_models[int(bbox['obj_id']) - 1].points
        mesh_pts = cad_models[int(bbox['obj_id']) - 1].points

        tmp.points = o3d.utility.Vector3dVector(mesh_pts)
        scene.paint_uniform_color([0, 0.651, 0.929])
        tmp.paint_uniform_color([1, 0.706, 0])
        # o3d.visualization.draw_geometries(
        #     [scene, tmp], window_name='gt vs est after icp')
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                                                     relative_rmse=0.000001,
                                                                     max_iteration=30)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            tmp, scene, 5, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria)
        R_icp = reg_p2p.transformation[0:3, 0:3].reshape(1, 9)[0]
        t_icp = reg_p2p.transformation[0:3, 3].T
        df.loc[(df["scene_id"] == int(scene_id)) & (
            df["im_id"] == int(im_id)) & (df["obj_id"] == int(obj_id)), 'R'] = np.array2string(R_icp, separator=" ")[1:-1].replace('\n', '')
        df.loc[(df["scene_id"] == int(scene_id)) & (
            df["im_id"] == int(im_id)) & (df["obj_id"] == int(obj_id)), 't'] = np.array2string(t_icp, separator=" ")[1:-1].replace('\n', '')
        tmp.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries(
            [scene, tmp], window_name='gt vs est after icp')
df.to_csv('./test.csv', quoting=csv.QUOTE_NONE,
          escapechar=",", index=False)
# print("icp: ", reg_p2p.transformation)


for index, row in df.iterrows():
    scene_id = df['scene_id'][index]
    im_id = df['im_id'][index]
    obj_id = df['obj_id'][index]
    R_est = df['R'][index]
    t_est = df['t'][index]
    # print(R_est)
