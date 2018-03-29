import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np


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

def classifier():
	vgg16 = models.vgg16(pretrained=True).cuda()
	for param in vgg16.parameters():
		# print param.shape
		param.requires_grad = False

	return vgg16

class ResBlock(nn.Module):
	def __init__(self,in_channels, out_channels, stride):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu1 = nn.ReLU(inplace = True)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
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

class UpsBlock(nn.Module):
	def __init__(self, h_channel, scale_factor = None, size = None):
		super(UpsBlock, self).__init__()
		if size is None:
			self.ups = nn.Upsample(scale_factor = scale_factor, mode = 'bilinear')
		else:
			self.ups = nn.Upsample(size = size, mode = 'bilinear')

		self.conv = nn.Conv2d(h_channel, h_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn = nn.BatchNorm2d(h_channel)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.ups(x)
		out = self.conv(out)
		out = self.bn(out)
		out = self.relu(out)

		return out 


class SRNet(nn.Module):
	def __init__(self, h_channel = 64):
		super(SRNet, self).__init__()
		self.conv1 = nn.Conv2d(3, h_channel, kernel_size = 9, stride = 1, padding = 4, bias = False)
		self.bn1 = nn.BatchNorm2d(h_channel)
		self.relu1 = nn.ReLU(inplace = True)

		self.res_block1 = ResBlock(h_channel,h_channel,1)
		self.res_block2 = ResBlock(h_channel,h_channel,1)
		self.res_block3 = ResBlock(h_channel,h_channel,1)
		self.res_block4 = ResBlock(h_channel,h_channel,1)

		self.ups2 = UpsBlock(h_channel, scale_factor = 2)

		self.ups3 = UpsBlock(h_channel, scale_factor = 2)

		self.conv4 = nn.Conv2d(h_channel, 3, kernel_size = 9, stride = 1, padding = 4, bias = False)
		self.bn4 = nn.BatchNorm2d(3)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.res_block1(out)
		out = self.res_block2(out)
		out = self.res_block3(out)
		out = self.res_block4(out)

		out = self.ups2(out)
		# out = self.ups3(out)

		out = self.conv4(out)
		out = self.bn4(out)

		return out




