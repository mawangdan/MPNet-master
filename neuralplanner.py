import argparse
import logging

import h5py
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset 
from model import MLP 
from torch.autograd import Variable 
import math
import time

def setup_logging(log_file):
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.FileHandler(log_file),  # 日志记录到文件
            logging.StreamHandler()  # 日志同时输出到控制台（可以选择性关闭）
        ]
    )
size=5.0
import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load trained model for path generation
mlp = MLP(60+7+7, 7) # simple @D
mlp.load_state_dict(torch.load('models/mlp_100_4000_PReLU_ae_dd_final.pkl',map_location=torch.device('cpu')))

if torch.cuda.is_available():
	mlp.cuda()


def load_urdf_model(urdf_path, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1]):
	model_id = p.loadURDF(urdf_path, basePosition=base_position, baseOrientation=base_orientation)
	return model_id


def check_collision(model_id1, obss):
	contact_points = p.getClosestPoints(model_id1, obss[0], distance=0)
	contact_points2 = p.getClosestPoints(model_id1, obss[1], distance=0)
	return len(contact_points) > 0 | len(contact_points2) >0



p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId = p.loadURDF("./franka_description/robots/panda_arm.urdf")  # KUKA_IIWA_URDF | DRAKE_IIWA_URDF

def IsInCollision(x,idx):



	# Disconnect from the physics server
	obss=[]
	for j in range(0, 2):
		if abs(sphere_radiis[idx][j] - 0.2) < 0.01:
			urdf_path="./medium_sphere.urdf"
			obss.append(load_urdf_model(urdf_path, base_position=sphere_centers[idx][j]))
		if abs(sphere_radiis[idx][j] - 0.1) < 0.01:
			urdf_path="./small_sphere.urdf"
			obss.append(load_urdf_model(urdf_path, base_position=sphere_centers[idx][j]))
		if abs(cube_sizes[idx][j] - 0.2) < 0.01:
			urdf_path="./medium_cube.urdf"
			obss.append(load_urdf_model(urdf_path, base_position=cube_centers[idx][j]))


	# Example usage
	for j in range(0,7):
		p.resetJointState(robotId, j, x[j])

	# Directly check for collisions without simulation steps
	collision_detected = check_collision(robotId, obss)
	return collision_detected


def steerTo (start, end, idx):

	DISCRETIZATION_STEP=0.01
	dists=np.zeros(7,dtype=np.float32)
	for i in range(0,7):
		dists[i] = end[i] - start[i]

	distTotal = 0.0
	for i in range(0,7):
		distTotal =distTotal+ dists[i]*dists[i]

	distTotal = math.sqrt(distTotal)
	if distTotal>0:
		incrementTotal = distTotal/DISCRETIZATION_STEP
		for i in range(0,7):
			dists[i] =dists[i]/incrementTotal



		numSegments = int(math.floor(incrementTotal))

		stateCurr = np.zeros(7,dtype=np.float32)
		for i in range(0,7):
			stateCurr[i] = start[i]
		for i in range(0,numSegments):

			if IsInCollision(stateCurr,idx):
				return 0

			for j in range(0,7):
				stateCurr[j] = stateCurr[j]+dists[j]


		if IsInCollision(end,idx):
			return 0


	return 1

# checks the feasibility of entire path including the path edges
def feasibility_check(path,idx):

	for i in range(0,len(path)-1):
		ind=steerTo(path[i],path[i+1],idx)
		if ind==0:
			return 0
	return 1


# checks the feasibility of path nodes only
def collision_check(path,idx):

	for i in range(0,len(path)):
		if IsInCollision(path[i],idx):
			return 0
	return 1

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,dataset,targets,seq,bs):
	bi=np.zeros((bs,18),dtype=np.float32)
	bt=np.zeros((bs,2),dtype=np.float32)
	k=0	
	for b in range(i,i+bs):
		bi[k]=dataset[seq[i]].flatten()
		bt[k]=targets[seq[i]].flatten()
		k=k+1
	return torch.from_numpy(bi),torch.from_numpy(bt)



def is_reaching_target(start,end):

	for i in range(0,7):
		if abs(start[i]-end[i]) > 1.0:
			return False
	return True

#lazy vertex contraction 
def lvc(path,idx):

	for i in range(0,len(path)-1):
		for j in range(len(path)-1,i+1,-1):
			ind=0
			ind=steerTo(path[i],path[j],idx)
			if ind==1:
				pc=[]
				for k in range(0,i+1):
					pc.append(path[k])
				for k in range(j,len(path)):
					pc.append(path[k])

				return lvc(pc,idx)
				
	return path

def re_iterate_path2(p,g,idx,obs):
	step=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			target_reached=False
			while (not target_reached) and itr<50 :
				new_path.append(st)
				itr=itr+1
				ip=torch.cat((obs,st,gl))
				ip=to_var(ip)
				st=mlp(ip)
				st=st.data.cpu()		
				target_reached=is_reaching_target(st,gl)
			if target_reached==False:
				return 0

	#new_path.append(g)
	return new_path

def replan_path(p,g,idx,obs):
	step=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			pA=[]
			pA.append(st)
			pB=[]
			pB.append(gl)
			target_reached=0
			tree=0
			while target_reached==0 and itr<50 :
				itr=itr+1
				if tree==0:
					ip1=torch.cat((obs,st,gl))
					ip1=to_var(ip1)
					st=mlp(ip1)
					st=st.data.cpu()
					pA.append(st)
					tree=1
				else:
					ip2=torch.cat((obs,gl,st))
					ip2=to_var(ip2)
					gl=mlp(ip2)
					gl=gl.data.cpu()
					pB.append(gl)
					tree=0		
				target_reached=steerTo(st, gl, idx)
			if target_reached==0:
				return 0
			else:
				for p1 in range(0,len(pA)):
					new_path.append(pA[p1])
				for p2 in range(len(pB)-1,-1,-1):
					new_path.append(pB[p2])

	return new_path	
sphere_centers,sphere_radiis,cube_centers,cube_sizes, obs_rep, m_length, solutions = load_test_dataset()
def main(args):

	# Create model directory
	global path
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	

	N=m_length.__len__()
	tp=0
	fp=0
	tot=[]
	logging.info("step: N="+str(N))
	for i in range(0,N):
		et=[]
		logging.info("step: i="+str(i))
		p1_ind=0
		p2_ind=0
		p_ind=0
		path=[]
		if m_length[i]>0:
			start=np.zeros(7,dtype=np.float32)
			goal=np.zeros(7,dtype=np.float32)
			for l in range(0,7):
				start[l]=solutions[i,0,l]

			for l in range(0,7):
				goal[l]=solutions[i,int(min(m_length[i],300)-1),l]
			#start and goal for bidirectional generation
			## starting point
			start1=torch.from_numpy(start)
			goal2=torch.from_numpy(start)
			##goal point
			goal1=torch.from_numpy(goal)
			start2=torch.from_numpy(goal)
			##obstacles
			obs=obs_rep[i]
			obs=torch.from_numpy(obs)
			##generated paths
			path1=[]
			path1.append(start1)
			path2=[]
			path2.append(start2)
			target_reached=0
			step=0
			path=[] # stores end2end path by concatenating path1 and path2
			tree=0
			tic = time.clock()
			while target_reached==0 and step<80 :
				step=step+1
				if tree==0:
					inp1=torch.cat((obs,start1,start2))
					inp1=to_var(inp1)
					start1=mlp(inp1)
					start1=start1.data.cpu()
					path1.append(start1)
					tree=1
				else:
					inp2=torch.cat((obs,start2,start1))
					inp2=to_var(inp2)
					start2=mlp(inp2)
					start2=start2.data.cpu()
					path2.append(start2)
					tree=0
				target_reached=steerTo(start1,start2,i)
			tp=tp+1

			if target_reached==1:
				for p1 in range(0,len(path1)):
					path.append(path1[p1])
				for p2 in range(len(path2)-1,-1,-1):
					path.append(path2[p2])


				path=lvc(path,i)
				indicator=feasibility_check(path,i)
				if indicator==1:
					toc = time.clock()
					t=toc-tic
					et.append(t)
					fp=fp+1
					# for pp in range(0,7):
					# 	print ("path[%d]:",pp)
					# 	for p in range(0,len(path)):
					# 		print (path[p][pp])
					# for pp in range(0,7):
					# 	print ("Actual path[%d]:",pp)
					# 	for p in range(0,int(min(m_length[i],300)-1)):
					# 		print (solutions[i,p,pp])
				else:
					sp=0
					indicator=0
					while indicator==0 and sp<10 and path !=0:
						sp=sp+1
						g=np.zeros(3,dtype=np.float32)
						g=torch.from_numpy(solutions[i,int(min(300,m_length[i])-1),:])
						path=replan_path(path,g,i,obs) #replanning at coarse level
						if path !=0:
							path=lvc(path,i)
							indicator=feasibility_check(path,i)

						if indicator==1:
							toc = time.clock()
							t=toc-tic
							et.append(t)
							fp=fp+1
							# if len(path)<20:
							# 	for pp in range(0, 7):
							# 		print("path[%d]:", pp)
							# 		for p in range(0, len(path)):
							# 			print(path[p][pp])
							# 	for pp in range(0, 7):
							# 		print("Actual path[%d]:", pp)
							# 		for p in range(0, int(min(m_length[i],300)-1)):
							# 			print(solutions[i, p, pp])
							# else:
							# 	print("path found, dont worry")
		tot.append(path)
	logging.info("save...")
	with h5py.File(f"test.hdf5", "w-") as f:
		dt = h5py.vlen_dtype(np.dtype('float32'))

		# 创建数据集并写入数据
		dset = f.create_dataset('tot', (len(tot),), dtype=dt)
		dset[:] = tot

		tmp = f.create_dataset("sphere_centers", (N, 2, 3))
		tmp[...] = sphere_centers[...]
		tmp = f.create_dataset("sphere_radiis", (N, 2))
		tmp[...] = sphere_radiis[...]
		tmp = f.create_dataset("cube_centers", (N, 2, 3))
		tmp[...] = cube_centers[...]
		tmp = f.create_dataset("cube_sizes", (N, 2))
		tmp[...] = cube_sizes[...]
	pickle.dump(tot, open("time_s2D_unseen_mlp.p", "wb" ))	


	print ("total paths")
	print (tp)
	print ("feasible paths")
	print (fp)

if __name__ == '__main__':
	setup_logging("log_test.log")
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--input_size', type=int , default=60+7+7, help='dimension of the input vector')
	parser.add_argument('--output_size', type=int , default=7, help='dimension of the input vector')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=28)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	args = parser.parse_args()
	print(args)
	main(args)


