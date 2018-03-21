import torch
from torch.autograd import Variable
import time
import torchvision.models as models
from torch.utils.serialization import load_lua
import torch.nn as nn
import h5py
import numpy as np
import argparse
import sys
import SRNet_model as model

def BatchGenerator(files,batch_size):
	for file in files:
		curr_data = h5py.File(file,'r')
		data = np.array(curr_data['data'],dtype='f')
		label = np.array(curr_data['label'],dtype='f')

		# print np.max(data), np.max(label)

		data = np.transpose(data,(0,3,1,2))
		label = np.transpose(label,(0,3,1,2))/255.0

		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])

		label = (label-mean[np.newaxis,:,np.newaxis,np.newaxis])/std[np.newaxis,:,np.newaxis,np.newaxis]

		# if border_mode=='valid':
		# 	label = crop(label,crop_size)

		for i in range((data.shape[0]-1)//batch_size + 1):
			# print data.shape
			data_bat = Variable(torch.FloatTensor(data[i*batch_size:(i+1)*batch_size])).cuda()
			label_bat = Variable(torch.FloatTensor(label[i*batch_size:(i+1)*batch_size])).cuda()
			# label_bat = Variable(torch.randn(64, 3, 200,200)).cuda()
			label_bat = model.feature_extractor(label_bat)

			label_bat = Variable(torch.FloatTensor(label_bat.data.cpu()), requires_grad = False).cuda()
			yield (data_bat, label_bat)

def save_checkpoint(model, optimizer,filename='default.pth'):
	torch.save({'model':model, 'optimizer':optimizer}, filename)

def train(data, label):

	model_name = sys.argv[1]
	epochs = int(sys.argv[2])


	srnet = model.SRNet_ext().cuda()

	N, D_in, D_out = 64, 50, 200
	path_train = '/home/sushobhan/Documents/data/fourier_ptychography/datasets/pcp_ptych/normalize/'
	home = "/home/sushobhan/Documents/data/fourier_ptychography/models/ptych_SR/"
	lr = 0.001
	batch_size = 64


	train_files = []
	val_files = []
	for i in range(1,80):
		train_files.append(path_train+str(i)+'.h5')

	for i in range(80,92):
		val_files.append(path_train+str(i)+'.h5')

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, srnet.parameters()), lr=lr)
	# train_generator = BatchGenerator(train_files,batch_size)
	loss_fn = nn.MSELoss().cuda()

	for i in range(epochs):
		j=0
		loss_arr = []

		# if i>10:
		# 	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, srnet.parameters()), lr=lr/10)
		start = time.time()
		train_generator = BatchGenerator(train_files,batch_size)
		for data,label in train_generator:
			x = data
			y = label
			optimizer.zero_grad()

			y_pred = srnet(x)

			loss = loss_fn(y_pred, y)
			print [i,j],
			loss_arr.append(loss.data[0])

			loss.backward()
			optimizer.step()
			j+=1
		print time.time() - start
		print np.mean(loss_arr)

	save_checkpoint(srnet.state_dict(),optimizer.state_dict(),home+model_name+'.pth')
	save_checkpoint(srnet,optimizer,home+model_name+'_mod.pth')


if __name__=='__main__':
	main()
