import numpy
import vtk as vt
import vtkgdcm as gdcm
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import pyplot as plt

class vtk_numpy():
	def __init__(self):
		vtk = gdcm.vtkGDCMImageReader()
		np = numpy.empty(0)		

	def transformar(self, arquivo):
		self.vtk.SetFileName(arquivo)
		self.vtk.Update()
		self.np = vtk_to_numpy(vtk.GetOutput().GetPointData().GetArray(0))
		self.np.shape = (512, 512)

	def imprimir(self):
		plt.imshow(self.np, cmap = "Greys_r")
		plt.show()

if __name__ == "__main__":
	import sys
	obj = vtk_numpy()
	obj.transformar(sys.argv[1])
	obj.imprimir()