# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:01:16 2020
Quasi Newton-Rhapson implementation

@author: Juan Camilo V
"""
import numpy as np
import math as mh
import matplotlib.pyplot as plt
import FEM_Hyperelastic as uel
import post_process as ppro

# Do not modify
ncoord=2
ndof=2
n=4
# Do not modify


## LOADING OF INPUT FILES
coords=np.loadtxt('nodesC', delimiter=',')
topo=np.loadtxt('elementsC', delimiter=',')
loads=np.loadtxt('loadsC', delimiter=',')
BC=np.loadtxt('bcelesC', delimiter=',')

## ASSIGNMENT OF VARIABLES FROM THE INPUT FILES TO NDIM ARRAYS
coords[:,0]=coords[:,0]-1
topo=topo-1 ### PARAMETER TO BE ABLE TO WORK WITH ABAQUS MESHES
topo=topo.astype(int)
if topo.ndim==1:
    nelem=1
else:
    nelem=topo.shape[0]
nnodes=coords.shape[0]
nBC=BC.shape[0]

## MATERIAL PROPERTIES. IN THIS VERSION, MODIFY MANUALLY
materialprops=np.array([1,10])

## INTIALIZATION OF DISPLACEMENTS VECTOR
U=np.zeros([nnodes*ndof,1])

## SOLVER PARAMETERS FOR THE SIMULATION
loadstep=5
tol=0.0001
maxit=30
relfac=1.


## NEWTON-RAPHSON SCHEME
for itera in range(loadstep):   
    actload=(itera+1)/(loadstep)   
    err=1.
    nit=0
    # print('Step', itera)    
    while nit<maxit and err>tol:
        nit=nit+1
        KG,STRESS=uel.assemblystiff(nelem,coords,materialprops,topo,U)        
        RG=uel.assemblyres(nelem,coords,materialprops,topo,U)
        FG=uel.nodalforces(coords,nelem,loads)        
        RHS=actload*FG-RG       
        for n in range(nBC):
            row=ndof*BC[n,0]+BC[n,1]
            row=row.astype(int)
            KG[row,:]=0.
            KG[row,row]=1.
            RHS[row]=actload*BC[n,2]-U[row]        
        dU=np.dot(np.linalg.inv(KG),RHS)       
        U=U+relfac*dU
        nU=U*U
        nU=sum(nU)
        err=dU*dU
        err=sum(err)
        err=mh.sqrt(err/nU)
        # print('Iteration',nit,'Correction',err,'Tolerance',tol)
  
print('Analysis Complete')   
## TRANSFORMATION OF STRESS AND STRAIN FROM IP TO NODES
E_nodes, S_nodes=uel.strain_nodes(nelem,coords,materialprops,topo,U)       
        
   

## Deformed Shape Plot
scale=0.5
dx=U[0:nnodes*ndof:2]
dy=U[1:nnodes*ndof:2]
# plt.figure(0)
# plt.plot(coords[:,1],coords[:,2],'o', color='black')
# dcoords=coords
# dcoords[:,1]=dcoords[:,1]+dx.T*scale
# dcoords[:,2]=dcoords[:,2]+dy.T*scale
# plt.plot(dcoords[:,1],dcoords[:,2],'o', color='red')
# plt.show()


## DISPLACEMENT FIELD PLOT
UV=np.concatenate((dx, dy),axis=1)
# A=ppro.fields_plot(topo, coords, UV, E_nodes=E_nodes, S_nodes=S_nodes)



### POST PROCESSING OF ABAQUS INFORMATION
# plt.figure(1)
UVAB=np.loadtxt('UABQCantilever')
SAB=np.loadtxt('SABQNodes')
EAB=np.loadtxt('EABQNodes')
UVAB=np.reshape(UVAB,(106,1)) ### ADJUST ACCORDING TO THE PROBLEM
SAB=np.reshape(SAB,(159,1)) 
EAB=np.reshape(EAB,(159,1)) 
dxAB=UVAB[0:53]
dyAB=UVAB[53:106]

SXAB=SAB[0:53]
SYAB=SAB[106:159]
SXYAB=SAB[53:106]
EXAB=EAB[0:53]
EYAB=EAB[106:159]
EXYAB=EAB[53:106]

UVAB=np.concatenate((dxAB, dyAB),axis=1)
SABF=np.concatenate((SXAB,SYAB,SXYAB),axis=1)
EABF=np.concatenate((EXAB,EYAB,EXYAB),axis=1)
# B=ppro.fields_plot(topo, coords, UVAB, E_nodes=EABF, S_nodes=SABF)


ERR=abs(UV-UVAB)
# C=ppro.fields_plot(topo, coords, ERR, E_nodes=None, S_nodes=None)

        
    
### STRESS Comparison CENTERLINE

SPx=STRESS[:,0,0]
SPy=STRESS[:,1,1]
SPxy=STRESS[:,0,1]

SAB=np.loadtxt('SABQCantilever')

SAB=SAB[4:-1:9]

SABx=SAB[0:10]
SABxy=SAB[10:20]
SABy=SAB[20:30]

X=np.linspace(0,10,num=10)

# plt.figure(102)
# plt.plot(X,SABx,'b',label='Abaqus')
# plt.plot(X,SPx,'--r',label='Python')
# plt.ylabel(r"$\sigma_x$")
# plt.xlabel('Element')
# plt.legend()
# E1=np.linalg.norm(SABx-SPx)

# plt.figure(103)
# plt.plot(X,SABxy,'b',label='Abaqus')
# plt.plot(X,SPxy,'--r',label='Python')
# plt.ylabel(r"$\sigma_{xy}$")
# plt.xlabel('Element')
# plt.legend()
# E2=np.linalg.norm(SABxy-SPxy)

# plt.figure(104)
# plt.plot(X,SABy,'b',label='Abaqus')
# plt.plot(X,SPy,'--r',label='Python')
# plt.ylabel(r"$\sigma_y$")
# plt.legend()
# plt.xlabel('Element')

E3=np.linalg.norm(SABy-SPy)