import vtk
import numpy as np
import vtkgdcm as gdcm
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import pyplot as plt

class vtk_numpy():
	def __init__(self):
		reader = gdcm.vtkGDCMImageReader()
		imagem = np.empty(0)

	def le_arquivo(self, arquivo):
		self.reader.SetFileName(arquivo)
		self.reader.Update()

	def transforma_arquivo(self):
		self.imagem = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))
		self.imagem.shape = (512, 512)

	def imprime_imagem(self):
		plt.imshow(self.imagem, cmap = "Greys_r")
		plt.show()

if __name__ == "__main__":
	import sys
	obj = vtk_numpy()
	obj.le_arquivo(sys.argv[1])
	obj.transforma_arquivo()
	obj.imprime_imagem()