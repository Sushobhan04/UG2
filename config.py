import numpy as np
import sys
import os

class Config(object):
	def __init__(self):
		super(Config, self).__init__()

		self.mode = "training" ## training/testing

		self.num_input_neurons = None
		self.num_output_neurons = None
		self.num_reservoir_neurons = None
		self.max_synapses = None
		self.threshold = None

		self.iter_per_unit = 1
		self.decay_rate = 1.0
		self.mean_intensity = 1.0
		self.update_batch_size = 1
		self.window_size = 2

		self.stdp_params = [1.0, 1.0]
		self.weight_update_rate = 0.01