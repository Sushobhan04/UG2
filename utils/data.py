import scipy.ndimage as im
import numpy as np
import os
import h5py
import torch
from torch.autograd import Variable
import pickle


def BatchGenerator(files, batch_size):
	for file in files:
		with h5py.File(file,'r') as curr_data:
			data = np.array(curr_data['data']).astype(np.float32)
			label = np.array(curr_data['label']).astype(np.float32)

		# print np.max(data), np.max(label)

		# data = data/255.0
		# label = label/255.0

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

def convert_to_torch_variable(tensor, cuda = True):
	if cuda:
		return Variable(torch.FloatTensor(tensor)).cuda()
	else:
		return Variable(torch.FloatTensor(tensor))

def create_h5(data, label, path, file_name):

	with h5py.File(os.path.join(path, file_name), 'w') as file:
		file.create_dataset("data", data = data)
		file.create_dataset("label", data = label)

def patchify(image, size):
	num_patches = [image.shape[1]//size[0], image.shape[2]//size[1]]
	patches = np.zeros((num_patches[0]*num_patches[1], 3, size[0], size[1]))

	for i in range(num_patches[0]):
		for j in range(num_patches[1]):
			patches[i*num_patches[1] + j] = image[:, i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]]

	return patches

def create_dataset(factor, num_images, patch_size, dataset_path, destination_path, dataset_name):

	folder = "image_SRF_" + str(factor)
	lr_suffix = "_".join(["SRF", str(factor), "LR"])
	hr_suffix = "_".join(["SRF", str(factor), "HR"])

	lr_stack = []
	hr_stack = []

	for i in range(1, 1+num_images):
		lr_img = im.imread(os.path.join(dataset_path, folder, "img_" + str(i).zfill(3) + "_" + lr_suffix + ".png"))
		hr_img = im.imread(os.path.join(dataset_path, folder, "img_" + str(i).zfill(3) + "_" + hr_suffix + ".png"))

		lr_img = np.transpose(lr_img, (2, 0, 1))
		hr_img = np.transpose(hr_img, (2, 0, 1))

		lr_stack.append(patchify(lr_img, size = patch_size))
		hr_stack.append(patchify(hr_img, size = patch_size*factor))

	lr_set = np.concatenate(lr_stack)
	hr_set = np.concatenate(hr_stack)

	create_h5(data = lr_set, label = hr_set, path = destination_path, file_name = dataset_name)

	print("data of shape ", lr_set.shape, "and label of shape ", hr_set.shape, " created of type", lr_set.dtype)


def parse_vatic_annotations(path, txt_filename):
	txt_file = open(os.path.join(path, txt_filename), "r")

	for line in txt_file:
		line = line.rstrip().split(" ")

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def parse_imagenet(path, file, img_size = 16):
	data_file = os.path.join(data_folder, file)

	d = unpickle(data_file)
	x = d['data']
	y = d['labels']
#     mean_image = d['mean']

	x = x/np.float32(255)
#     mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0
	y = [i-1 for i in y]
	data_size = x.shape[0]

#     x -= mean_image

	img_size2 = img_size * img_size

	x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
	x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
	
	return x,y	

