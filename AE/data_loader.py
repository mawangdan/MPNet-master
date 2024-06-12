import h5py
import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import trimesh
from PIL import Image
import os.path
import random
import pybullet as p
import pybullet_data
import open3d as o3d

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId = p.loadURDF("./franka_description/robots/panda_arm.urdf")  # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])
def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)
def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])
def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return (point, quat_from_euler(euler))

CLIENT = 0
def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)


def sample_points_from_sphere(center, radius, num_points):
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    r = radius * np.cbrt(np.random.rand(num_points))
    x = r * np.sin(phi) * np.cos(theta) + center[0]
    y = r * np.sin(phi) * np.sin(theta) + center[1]
    z = r * np.cos(phi) + center[2]
    return np.vstack((x, y, z)).T


def sample_points_from_box(center, half_extents, num_points):
    x = np.random.uniform(center[0] - half_extents[0], center[0] + half_extents[0], num_points)
    y = np.random.uniform(center[1] - half_extents[1], center[1] + half_extents[1], num_points)
    z = np.random.uniform(center[2] - half_extents[2], center[2] + half_extents[2], num_points)
    return np.vstack((x, y, z)).T


def sample_points_from_cylinder(center, radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(center[2] - height / 2, center[2] + height / 2, num_points)
    r = radius * np.sqrt(np.random.rand(num_points))
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.vstack((x, y, z)).T


def get_point_cloud_from_urdf(urdf_path, num_points,center):
    # Connect to PyBullet in DIRECT mode (no GUI)
    p.connect(p.DIRECT)

    # Load the URDF model
    robot_id = p.loadURDF(urdf_path)
    set_pose(robot_id,
             Pose(Point(x=center[0], y=center[1], z=center[2]), Euler()))
    # Get collision shape data
    collision_shape_data = p.getVisualShapeData(robot_id)

    # Extract points from the geometry
    all_points = []
    for shape in collision_shape_data:
        geom_type = shape[2]
        origin = shape[5]
        if geom_type == p.GEOM_SPHERE:
            radius = shape[3][0]
            points = sample_points_from_sphere(origin, radius, num_points // len(collision_shape_data))
        elif geom_type == p.GEOM_BOX:
            half_extents = shape[3]
            points = sample_points_from_box(origin, half_extents, num_points // len(collision_shape_data))
        elif geom_type == p.GEOM_CYLINDER:
            radius = shape[3][0]
            height = shape[3][1]
            points = sample_points_from_cylinder(origin, radius, height, num_points // len(collision_shape_data))
        elif geom_type == p.GEOM_MESH:
            vertices = p.getMeshData(robot_id, shape[1])[1]
            triangles = p.getMeshData(robot_id, shape[1])[3]
            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                             triangles=o3d.utility.Vector3iVector(triangles))
            sampled_points = mesh.sample_points_uniformly(number_of_points=num_points // len(collision_shape_data))
            points = np.asarray(sampled_points.points)
        else:
            continue

        all_points.append(points)

    # Combine all points
    combined_points = np.vstack(all_points)

    # Randomly sample required number of points
    if len(combined_points) > num_points:
        indices = np.random.choice(len(combined_points), num_points, replace=False)
        sampled_points = combined_points[indices]
    else:
        sampled_points = combined_points

    p.disconnect()
    return sampled_points

def save_point_cloud(NP=1800):


    with h5py.File(f"./test_data-10.hdf5", "r") as f:
        m_length = f["m_length"][:]
        N = int(m_length.__len__())
        obstacles_point_list = np.zeros((N, 1400 * 3), dtype=np.float32)
        solutions = f["solutions"][:]
        sphere_centers = f["sphere_centers"][:]
        sphere_radiis = f["sphere_radiis"][:]
        cube_centers = f["cube_centers"][:]
        cube_sizes = f["cube_sizes"][:]
        # 获取机械臂的关节数
        num_joints = 7
        end_effector_positions=np.zeros((N,300, 3), dtype=np.float32)
        # 打印所有关节信息
        for i in range(num_joints):
            print(p.getJointInfo(robotId, i))
        for i in range(0,m_length.__len__()):
            for k in range(0,min(300,int(m_length[i][0]))):
                for j in range(num_joints):
                    p.resetJointState(robotId, j, solutions[i][k][j])
                # 获取末端执行器的位置和方向
                link_state = p.getLinkState(robotId, num_joints - 1)
                end_effector_positions[i][k][0]= link_state[4][0] # 末端执行器的位置
                end_effector_positions[i][k][1]= link_state[4][1] # 末端执行器的位置
                end_effector_positions[i][k][2]= link_state[4][2] # 末端执行器的位置

        for i in range(0,m_length.__len__()):
            print(i)
            obstacles_point=[]


            for j in range(0,2):
                if abs(sphere_radiis[i][j] - 0.2) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../medium_sphere.urdf", 700, sphere_centers[i][j]))
                if abs(sphere_radiis[i][j] - 0.1) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../small_sphere.urdf", 700, sphere_centers[i][j]))
                if abs(cube_sizes[i][j] - 0.2) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../medium_cube.urdf", 700, cube_centers[i][j]))

            obstacles_point_list[i]=np.concatenate((obstacles_point[0], obstacles_point[1]), axis=1).flatten()
        with h5py.File(f"test_point_cloud.hdf5", "w-") as f:
            tmp = f.create_dataset("obstacles_point_list", (N, 1400*3))
            tmp[...]=obstacles_point_list[...]
            tmp = f.create_dataset("solutions", (N, 300,7))
            tmp[...] = solutions[...]
            tmp = f.create_dataset("m_length", (N, 1))
            tmp[...] = m_length[...]
            tmp = f.create_dataset("end_effector_positions", (N,300, 3))
            tmp[...] = end_effector_positions[...]
            tmp = f.create_dataset("sphere_centers", (N, 2, 3))
            tmp[...] = sphere_centers[...]
            tmp = f.create_dataset("sphere_radiis", (N, 2))
            tmp[...] = sphere_radiis[...]
            tmp = f.create_dataset("cube_centers", (N, 2, 3))
            tmp[...] = cube_centers[...]
            tmp = f.create_dataset("cube_sizes", (N, 2))
            tmp[...] = cube_sizes[...]



def load_dataset(N=6075,NP=1800):
    obstacles_point_list = np.zeros((N, 1400*3), dtype=np.float32)
    with h5py.File(f"train_data.hdf5", "r") as f:
        m_length = f["m_length"][:]
        solutions = f["solutions"][:]
        sphere_centers = f["sphere_centers"][:]
        sphere_radiis = f["sphere_radiis"][:]
        cube_centers = f["cube_centers"][:]
        cube_sizes = f["cube_sizes"][:]
        for i in range(0,m_length.__len__()):
            print(i)
            obstacles_point=[]


            for j in range(0,2):
                if abs(sphere_radiis[i][j] - 0.2) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../medium_sphere.urdf", 700, sphere_centers[i][j]))
                if abs(sphere_radiis[i][j] - 0.1) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../small_sphere.urdf", 700, sphere_centers[i][j]))
                if abs(cube_sizes[i][j] - 0.2) <0.01:
                    obstacles_point.append(
                        get_point_cloud_from_urdf("../medium_cube.urdf", 700, cube_centers[i][j]))

            obstacles_point_list[i]=np.concatenate((obstacles_point[0], obstacles_point[1]), axis=1).flatten()
    return 	obstacles_point_list
def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
if __name__ == '__main__':
    save_point_cloud()