import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np
import time
import h5py
import os
from UG2.UG2.src import image_utils, data_utils

def BatchGenerator(files, batch_size):
	for file in files:
		curr_data = h5py.File(file,'r')
		data = np.array(curr_data['data']).astype(np.float32)
		label = np.array(curr_data['label']).astype(np.float32)
		curr_data.close()

		# print np.max(data), np.max(label)

		data = data/255.0
		label = label/255.0

		# mean = np.array([0.485, 0.456, 0.406])
		# std = np.array([0.229, 0.224, 0.225])

		# label = (label-mean[np.newaxis,:,np.newaxis,np.newaxis])/std[np.newaxis,:,np.newaxis,np.newaxis]

		# if border_mode=='valid':
		# 	label = crop(label,crop_size)

		for i in range((data.shape[0]-1)//batch_size + 1):
			# print data.shape
			data_bat = convert_to_torch_variable(data[i*batch_size:(i+1)*batch_size])
			label_bat = convert_to_torch_variable(label[i*batch_size:(i+1)*batch_size])

			yield (data_bat, label_bat)

class GenerateDataset(Dataset):
	def __init__(self, size, length):
		super(GenerateDataset, self).__init__()

		self.len = length


def convert_to_torch_variable(tensor, cuda = True):
	if cuda:
		return Variable(torch.FloatTensor(tensor)).cuda()
	else:
		return Variable(torch.FloatTensor(tensor))

def save_model(model, optimizer, path = "/", filename = 'check_point.pth'):
	torch.save({'model':model, 'optimizer':optimizer}, os.path.join(path, filename))


def train(config):

	self.forward = nn.DataParallel(self).forward
	if config.cuda:
		self.cuda()

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = config.lr)
	loss_fn = nn.MSELoss().cuda()

	for i in range(config.epochs):
		loss_arr = []

		batch_generator = BatchGenerator(config.train_files, config.batch_size)

		start = time.time()
		for x, y in batch_generator:
			optimizer.zero_grad()

			y_pred = self.forward(x)

			loss = loss_fn(self.feature_extractor(y_pred), self.feature_extractor(y))
			loss_arr.append(loss.data[0])

			loss.backward()
			optimizer.step()

		if i%config.print_step == 0:
			print("time: ", time.time() - start, " Error: ", np.mean(loss_arr))

		if i%config.checkpoint == 0:
			save_model(self.state_dict(), optimizer.state_dict(), path = config.model_save_path)
			print("saved checkpoint at epoch: ", i)


	save_model(self.state_dict(), optimizer.state_dict(), path = config.model_save_path, filename = config.model_name)
	print("Model saved as: ", config.model_name)

def test(self, img, hist_eq = True):
	_img = convert_to_torch_variable(img)

	out = self.forward(_img)
	out = out.data.cpu().numpy()

	if hist_eq:
		out = image_utils.hist_match(out, img)

	return out

def load_model(self, path, name):
	model = torch.load(os.path.join(path, name))["model"]
	self.load_state_dict(model)





