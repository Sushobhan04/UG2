import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np

def weights_init(m):
	classname = m.__class__.__name__
	if isinstance(m, nn.Conv2d):
		nn.init.dirac(m.weight.data)


def feat_ext(ext_type = "vgg16", cuda = True):
	temp_model = None
	model = None
	num_layers = 0

	if ext_type == "vgg16":
		temp_model = models.vgg16(pretrained=True)
		num_layers = 9

	elif ext_type == "vgg16_bn":
		temp_model = models.vgg16_bn(pretrained=True)
		# print(temp_model)
		num_layers = 13

	for param in temp_model.parameters():
		param.requires_grad = False

	model = torch.nn.Sequential(*(temp_model.features[i] for i in range(num_layers)))

	if cuda:
		model.cuda()

	return model

def pretrained_classifier(classifier_type, cuda = True):
	classifier = None

	if classifier_type == "vgg16":
		classifier = models.vgg16(pretrained=True)

	elif classifier_type == "vgg16_bn":
		classifier = models.vgg16_bn(pretrained=True)
		
	elif classifier_type == "vgg19":
		classifier = models.vgg19(pretrained=True)

	elif classifier_type == "resnet50":
		classifier = models.resnet50(pretrained=True)

	for param in classifier.parameters():
			param.requires_grad = False


	if cuda:
		classifier.cuda()

	return classifier

class Classifier(nn.Module):
	def __init__(self, classifier, size):
		super(Classifier, self).__init__()

		self.ups = nn.Upsample(size = size, mode = 'bilinear')
		self.classifier = classifier
		self.softmax = nn.Softmax(dim = 1)


	def forward(self, x, box = None):
		if box is not None:
			# print(x.shape, box)
			x = x[:, :, box[0,1]:box[0,3], box[0,0]:box[0,2]]
			
		out = self.ups(x)
		out = self.classifier(out)
		out = self.softmax(out)

		return out

class ResBlock(nn.Module):
	def __init__(self,in_channels, out_channels, stride):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu1 = nn.ReLU(inplace = True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
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

class ConvBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(ConvBlock, self).__init__()

		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)

		return out


class SRNet(nn.Module):
	def __init__(self, h_channel = 64, num_resblock = 4):
		super(SRNet, self).__init__()

		self.conv1 = nn.Conv2d(3, h_channel, kernel_size = 9, stride = 1, padding = 4, bias = False)
		self.bn1 = nn.BatchNorm2d(h_channel)
		self.relu1 = nn.ReLU(inplace = True)

		self.res_blocks = []

		# for i in range(num_resblock):
		# 	self.res_blocks.append(ResBlock(h_channel,h_channel,1))

		self.res_block1 = ResBlock(h_channel,h_channel,1)
		self.res_block2 = ResBlock(h_channel,h_channel,1)
		self.res_block3 = ResBlock(h_channel,h_channel,1)
		self.res_block4 = ResBlock(h_channel,h_channel,1)


		self.conv3 = ConvBlock(h_channel, h_channel)

		self.conv4 = nn.Conv2d(h_channel, 3, kernel_size = 9, stride = 1, padding = 4, bias = False)
		self.bn4 = nn.BatchNorm2d(3)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		# for res_block in self.res_blocks:
		# 	out = res_block(out)

		out = self.res_block1(out)
		out = self.res_block2(out)
		out = self.res_block3(out)
		out = self.res_block4(out)

		out = self.conv3(out)

		out = self.conv4(out)
		out = self.bn4(out)

		return out