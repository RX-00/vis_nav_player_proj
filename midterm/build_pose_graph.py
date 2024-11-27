import numpy as np
from scipy.spatial.transform import Rotation as R
import json


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_matrix()

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_matrix()



pose_graph = {}

with open('/Users/denismbeyakola/Desktop/vis_nav_player_proj/colmap_2/sparse/1/images.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        parts = line.split()
        image_name = parts[-1].split(".")[0]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        translation_vector = np.array([tx, ty, tz])

        pose_graph[image_name] = {
            'R': rotation_matrix,
            't': translation_vector,
            # 'name': parts[9]  # Image file name
        }

print(len(pose_graph.keys()))
with open('pose_graph.json', 'w') as f:
    json.dump(pose_graph, f, indent=4, default=lambda x: x.tolist())