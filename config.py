import numpy as np
import sys
import os

class Config(object):
	def __init__(self):
		super(Config, self).__init__()

		self.epochs = 1 ## training/testing
		self.cuda = True
		self.data_parallel = True
		self.lr = 0.001
		self.train_files = None
		self.batch_size = 1
		self.print_step = 1
		self.checkpoint = 1
		self.resume_training_flag = False
		self.resume_training_path = "/"
		self.resume_model_name = None

		self.model_path = "/"
		self.model_name = None