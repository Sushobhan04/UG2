import scipy.ndimage as im
import numpy as np
import os
import h5py
import torch
from torch.autograd import Variable
import pickle
from torch.utils.data import Dataset, DataLoader
import pickle
from UG2.utils import image as image_utils

class DatasetFromFile(Dataset):
	def __init__(self, path, data_file, img_size, data_format = "h5", transform = None):
		super(DatasetFromFile, self).__init__()
		self.transform = transform
		self.img_size = img_size

		if data_format == "h5":
			self.data, self.label = self.load_h5_data(path, data_file)
		elif data_format == "binary":
			self.data, self.label = self.load_binary_data(path, data_file)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return {"data": self.data[idx], "label": self.label[idx]}

	def load_binary_data(self, path, data_file):
		data_file = os.path.join(path, data_file)

		d = unpickle(data_file)
		x = d['data']
		y = d['labels']
	#     mean_image = d['mean']

		x = x/np.float32(255)
	#     mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0

		y_one_hot = np.zeros((len(y), 1000), dtype = np.float32)

		for i in range(len(y)):
			y_one_hot[i, y[i]-1] = 1.0
		data_size = x.shape[0]

	#     x -= mean_image

		img_size2 = self.img_size * self.img_size

		x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
		x = x.reshape((x.shape[0], self.img_size, self.img_size, 3)).transpose(0, 3, 1, 2)
		
		return x,y_one_hot

	def load_h5_data(self, path, data_file, dtype = "uint8"):

		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			data = np.array(curr_data['data']).astype(np.float32)
			label = np.array(curr_data['label']).astype(np.float32)

		# print np.max(data), np.max(label)

		if dtype == "uint8":
			data = data/255.0
			label = label/255.0

		return data, label

class ImagenetDataset(Dataset):
	def __init__(self, path, data_file, img_size, data_format = "h5", transform = None):
		super(ImagenetDataset, self).__init__()
		self.transform = transform
		self.img_size = img_size

def unpickle(file):
	with open(file, 'rb') as fo:
	  dict = pickle.load(fo)
	return dict

def convert_to_torch_tensor(tensor, cuda = True, from_numpy = True, requires_grad = False):
	if from_numpy:
		tensor = torch.FloatTensor(tensor)

	tensor = Variable(tensor)

	if cuda:
		tensor.cuda()

	if requires_grad:
		tensor.requires_grad = True

	return tensor

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

def create_bsd_dataset(factor, num_images, patch_size, dataset_path, destination_path, dataset_name):

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

def create_dataset(data_source_path, source_name_files, image_format, destination_path, dataset_name, num_images, patch_size, testing_fraction, blur_parameters):

	hr_image = []
	for i in range(num_images):
		image_name          = os.path.join(data_source_path,source_name_files[i]+image_format)
		hr_image.append(im.imread(image_name))   

	gen_data, gen_label = image_utils.blur_images(hr_image, blur_parameters["nTK"] ,blur_parameters["scale_factor"], blur_parameters["flags"], blur_parameters["gaussian_blur_range"])

	lr_stack = []
	hr_stack = []

	for i in range(1, 1+num_images):
		hr_img = gen_label[i]
		lr_img = gen_data[i]

		lr_stack.append(patchify(lr_img, size = patch_size))
		hr_stack.append(patchify(hr_img, size = patch_size*blur_parameters["scale_factor"]))

	lr_set = np.concatenate(lr_stack)
	hr_set = np.concatenate(hr_stack)

	number_training_images = int(testing_fraction*lr_set.shape[0])

	lr_set_training = lr_set[1:number_training_images]
	hr_set_training = hr_set[1:number_training_images]

	lr_set_testing  = lr_set[number_training_images+1:]
	hr_set_testing  = hr_set[number_training_images+1:]

	create_h5(data = lr_set_training, label = hr_set_training, path = destination_path, file_name = dataset_name+".h5")
	create_h5(data = lr_set_testing, label = hr_set_testing, path = destination_path, file_name = dataset_name+"_testing.h5")
	print("data of shape ", lr_set_training.shape, "and label of shape ", hr_set_training.shape, " created of type", lr_set_training.dtype)


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
           
