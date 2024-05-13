# compute fps (farthest point sampling) for models
import os.path as osp
import sys
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
import mmcv
from lib.pysixd import inout, misc
import ref
import open3d as o3d
from core.utils.data_utils import get_fps_and_center


ref_key = "delta_full"
data_ref = ref.__dict__[ref_key]

model_dir = data_ref.model_dir
id2obj = data_ref.id2obj


def main():
    vertex_scale = 0.001
    fps_dict = {}
    for obj_id in tqdm(id2obj):
        print(obj_id)
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        fps_dict[str(obj_id)] = {}
        fps_dict[str(obj_id)]["fps4_and_center"] = get_fps_and_center(model["pts"], num_fps=4, init_center=True)
        fps_dict[str(obj_id)]["fps8_and_center"] = get_fps_and_center(model["pts"], num_fps=8, init_center=True)
        fps_dict[str(obj_id)]["fps12_and_center"] = get_fps_and_center(model["pts"], num_fps=12, init_center=True)
        fps_dict[str(obj_id)]["fps16_and_center"] = get_fps_and_center(model["pts"], num_fps=16, init_center=True)
        fps_dict[str(obj_id)]["fps20_and_center"] = get_fps_and_center(model["pts"], num_fps=20, init_center=True)
        fps_dict[str(obj_id)]["fps32_and_center"] = get_fps_and_center(model["pts"], num_fps=32, init_center=True)
        fps_dict[str(obj_id)]["fps64_and_center"] = get_fps_and_center(model["pts"], num_fps=64, init_center=True)
        fps_dict[str(obj_id)]["fps128_and_center"] = get_fps_and_center(model["pts"], num_fps=128, init_center=True)
        fps_dict[str(obj_id)]["fps256_and_center"] = get_fps_and_center(model["pts"], num_fps=256, init_center=True)
        pcdd = fps_dict[str(obj_id)]["fps256_and_center"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcdd)
        import numpy as np
        pcd.paint_uniform_color(np.array([1,0,0]))
        #o3d.visualization.draw_geometries,([pcd],)
                                 
                                 
        o3d.io.write_point_cloud(osp.join(model_dir, "fps_models/" + str(obj_id)+".ply"), pcd)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(model["pts"])
        #o3d.visualization.draw_geometries([pcd],)
                                 
                                 

    save_path = osp.join(model_dir, "fps_points.pkl")
    #mmcv.dump(fps_dict, save_path)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
