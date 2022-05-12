import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def inner_product(Q, x, y):
	return x.T @ Q @ y

def norm_from_inner_product(Q):
	def norm(x):
		return np.sqrt(inner_product(Q, x, x))
	return norm

def lp_norm(p):
	def norm(x):
		return np.power(np.sum(np.power(np.abs(x), p)), 1.0 / p)
	return norm

def normalize(norm, x):
	return x / norm(x)

def plot_disk(norm, title):
	dim = 150
	center = (dim - 1.0) / 2.0
	stretch = 7.0
	img = np.ones((dim, dim)) * 255
	
	for i in range(dim):
		for j in range(dim):
			x = np.empty((2, 1))
			x[0, 0] = i
			x[1, 0] = j
			x = stretch * (x - (np.ones_like(x) * center)) / (dim - 1.0)
			x = normalize(norm, x)
			x = (x * (dim - 1.0) / stretch) + (np.ones_like(x) * center)
			img[int(x[0, 0]), int(x[1, 0])] = 0

	cv2.imwrite("outputs/" + title + "_Disk.png", img)


Q1 = np.array([[9, 0],
				[0, 1]])

Q2 = np.array([[9, 2],
				[2, 1]])

Q3 = np.array([[9, -2],
				[-2, 1]])

plot_disk(norm_from_inner_product(Q1), "Q1")
plot_disk(norm_from_inner_product(Q2), "Q2")
plot_disk(norm_from_inner_product(Q3), "Q3")
plot_disk(lp_norm(1), "L1")
plot_disk(lp_norm(2), "L2")