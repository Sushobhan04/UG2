import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import numpy as np
import time
import h5py
import os

def BatchGenerator(files, batch_size):
	for file in files:
		curr_data = h5py.File(file,'r')
		data = np.array(curr_data['data']).astype(np.float32)
		label = np.array(curr_data['label']).astype(np.float32)

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
			data_bat = Variable(torch.FloatTensor(data[i*batch_size:(i+1)*batch_size])).cuda()
			label_bat = Variable(torch.FloatTensor(label[i*batch_size:(i+1)*batch_size])).cuda()

			yield (data_bat, label_bat)

def convert_to_torch_variable(tensor, cuda = True):
	if cuda:
		return Variable(torch.FloatTensor(tensor)).cuda()
	else:
		return Variable(torch.FloatTensor(tensor))

def save_model(model, optimizer, path = "/", filename = 'check_point.pth'):
	torch.save({'model':model, 'optimizer':optimizer}, os.path.join(path, filename))

def feat_ext():
	vgg16 = models.vgg16(pretrained=True).cuda()
	for param in vgg16.parameters():
		# print param.shape
		param.requires_grad = False

	# label_bat = Variable(torch.randn(64, 3, 200,200)).cuda()
	# print vgg16
	model = torch.nn.Sequential(*(vgg16.features[i] for i in range(9)))
	# print model(label_bat)[0,0,0,0]
	return model

class ResBlock(nn.Module):
	def __init__(self,in_channels, out_channels, stride):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = x
		out = self.conv1(residual)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)

		out += x
		return out


class SRNet(nn.Module):
	def __init__(self, h_channel = 64):
		super(SRNet, self).__init__()
		self.conv1 = nn.Conv2d(3, h_channel, kernel_size = 9, stride = 1, padding = 4, bias=False)
		self.bn1 = nn.BatchNorm2d(h_channel)
		self.relu1 = nn.ReLU(inplace=True)

		self.res_block1 = ResBlock(h_channel,h_channel,1)
		self.res_block2 = ResBlock(h_channel,h_channel,1)
		self.res_block3 = ResBlock(h_channel,h_channel,1)
		self.res_block4 = ResBlock(h_channel,h_channel,1)

		self.ups2 = nn.Upsample(scale_factor = 2, mode='bilinear')
		self.conv2 = nn.Conv2d(h_channel, h_channel, kernel_size = 3, stride = 1, padding = 1, bias=False)
		self.bn2 = nn.BatchNorm2d(h_channel)
		self.relu2 = nn.ReLU(inplace=True)

		self.ups3 = nn.Upsample(scale_factor = 2, mode='bilinear')
		self.conv3 = nn.Conv2d(h_channel, h_channel, kernel_size = 3, stride = 1, padding = 1, bias=False)
		self.bn3 = nn.BatchNorm2d(h_channel)
		self.relu3 = nn.ReLU(inplace=True)

		self.conv4 = nn.Conv2d(h_channel, 3, kernel_size = 9, stride = 1, padding = 4, bias=False)
		self.bn4 = nn.BatchNorm2d(3)

		self.feature_extractor = feat_ext()


	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.res_block1(out)
		out = self.res_block2(out)
		out = self.res_block3(out)
		out = self.res_block4(out)

		out = self.ups2(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		# out = self.ups3(out)
		# out = self.conv3(out)
		# out = self.bn3(out)
		# out = self.relu3(out)

		out = self.conv4(out)
		out = self.bn4(out)

		return out

	def train(self, config):
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

	def test(self, img):
		img = convert_to_torch_variable(img)
		out = self.forward(img)

		return out





