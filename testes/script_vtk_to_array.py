import vtk
import numpy as np
import vtkgdcm as gdcm
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import pyplot as plt

def le_arquivo(arquivo):
	reader = gdcm.vtkGDCMImageReader()
	reader.SetFileName(arquivo)
	reader.Update()
	return reader

def transforma_arquivo(reader):
	imagem = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))
	imagem.shape = (512, 512)
	return imagem

def imprime_imagem(imagem):
	plt.imshow(imagem, cmap = "Greys_r")
	plt.show()

if __name__ == "__main__":
	import sys
	reader = le_arquivo(sys.argv[1])
	imagem = transforma_arquivo(reader)
	imprime_imagem(imagem)
