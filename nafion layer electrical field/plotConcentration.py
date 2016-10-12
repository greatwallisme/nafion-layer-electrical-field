import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File Name
fmMeshR = "Membrane mesh-R.txt"
fmMeshZ = "Membrane mesh-Z.txt"
fsMeshR = "Solution mesh-R.txt"
fsMeshZ = "Solution mesh-Z.txt"

fmCation = "mCation Concentration.txt"
fmProduct = "mProduct Concentration.txt"
fmReactant = "mReactant Concentration.txt"
fmPotential = "mPotential Distribution.txt"

fsAnion = "sAnion Concentration.txt"
fsCation = "sCation Concentration.txt"
fsProduct = "sProduct Concentration.txt"
fsReactant = "sReactant Concentration.txt"
fsPotential = "sPotential Distribution.txt"

# Import Data
mMeshR = np.genfromtxt(fmMeshR)*1e4
mMeshZ = np.genfromtxt(fmMeshZ)*1e4
sMeshR = np.genfromtxt(fsMeshR)*1e4
sMeshZ = np.genfromtxt(fsMeshZ)*1e4

mCation = np.genfromtxt(fmCation)
mProduct = np.genfromtxt(fmProduct)
mReactant = np.genfromtxt(fmReactant)
mPotential = np.genfromtxt(fmPotential)*1e3

sAnion = np.genfromtxt(fsAnion)
sCation = np.genfromtxt(fsCation)
sProduct = np.genfromtxt(fsProduct)
sReactant = np.genfromtxt(fsReactant)
sPotential = np.genfromtxt(fsPotential)*1e3

# Plot Data
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
ax1.plot_surface(mMeshR, mMeshZ, mPotential)
ax1.set_xlabel(r'R/$\mu$m')
ax1.set_ylabel(r'Z/$\mu$m')
ax1.set_zlabel(r'E/mV')
ax1.set_title('Membrane Phase Potential Distribution')

ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
ax2.plot_surface(sMeshR, sMeshZ, sPotential)
ax2.set_xlabel(r'R/$\mu$m')
ax2.set_ylabel(r'Z/$\mu$m')
ax2.set_zlabel(r'E/mV')
ax2.set_title('Solution Phase Potential Distribution')
plt.show()
