# -*- coding: utf-8 -*-
"""

Finite Element code for Hyperelasticity with 2D elements Plane Strain
Created on Mon Sep 21 13:11:30 2020

@author: Juan Camilo V
"""



## No distribuited Loads, Just Nodal Loads


import numpy as np
import math as mh
import matplotlib.pyplot as plt


### This function calculates the Tangent Stiffness Matrix
def materialstiffness(ndof,ncoord,B,J,materialprops):
    
    mu1=materialprops[0]
    K1=materialprops[1]
    
    dl=np.array([[1,0,0],[0,1,0],[0,0,1]]) ## Kronecker Delta
    
    C=np.zeros([2,2,2,2])
    # print('B',B,J)
    
    if (ncoord==2):
        Bqq=B[0,0]+B[1,1]+1.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        C[i,j,k,l]=(mu1*(dl[i,k]*B[j,l]+B[i,l]*dl[j,k]
                                    -(2/3)*(B[i,j]*dl[k,l]+dl[i,j]*B[k,l])
                                    + (2/3)*Bqq*dl[i,j]*dl[k,l]/3)/J**(2/3)
                                    + K1*(2*J-1)*J*dl[i,j]*dl[k,l])
    return C


#   Computes stress sigma_{ij} given B_{ij}

def Kirchhoffstress(ndof,ncoord,B,J,materialprops):
    stress=np.zeros([ndof, ncoord])
    dl=np.array([[1,0,0],[0,1,0],[0,0,1]]) ## Kronecker Delta
    mu1 = materialprops[0]
    K1 = materialprops[1]
    
    Bkk=B.trace()
    if (ndof==2):
        Bkk=Bkk+1
    for i in range(ndof):
        for j in range(ncoord):
            stress[i,j]=mu1*(B[i,j] - Bkk*dl[i,j]/3.)/J**(2/3)+K1*J*(J-1)*dl[i,j]
            #Kirchhofstress=CauchyStress*J
    return stress



#   The number of integration points in this case is 9, and 8-noded Elements

## Position of the Integration points
def GSPT():
    ndof=2
    nIP=9
    xi=np.zeros([ndof,nIP]) ## 2D 9-IP
    xi[0,0] = -mh.sqrt(3/5)
    xi[1,0] = xi[0,0]
    xi[0,1] = 0.0
    xi[1,1] = xi[0,0]
    xi[0,2] = -xi[0,0]
    xi[1,2] = xi[0,0]
    xi[0,3] = xi[0,0]
    xi[1,3] = 0.0
    xi[0,4] = 0.0
    xi[1,4] = 0.0
    xi[0,5] = -xi[0,0]
    xi[1,5] = 0.0
    xi[0,6] = xi[0,0]
    xi[1,6] = -xi[0,0]
    xi[0,7] = 0.
    xi[1,7] = -xi[0,0]
    xi[0,8] = -xi[0,0]
    xi[1,8] = -xi[0,0]
    
    return xi

def GPSW():
    nIP=9
    
    w=np.zeros([nIP,1])
    W1D=np.array([5/9,8/9,5/9])
    for j in range(3):
        for i in range(3):
            n=3*(j)+i
            w[n]=W1D[i]*W1D[j]
    return w

# Calculation of the shape functions

def shapefunc(xi):
    
    N=np.zeros([8,1])
    
    N[0] = -0.25*(1.-xi[0])*(1.-xi[1])*(1.+xi[0]+xi[1])
    N[1] = 0.25*(1.+xi[0])*(1.-xi[1])*(xi[0]-xi[1]-1.)
    N[2] = 0.25*(1.+xi[0])*(1.+xi[1])*(xi[0]+xi[1]-1.)
    N[3] = 0.25*(1.-xi[0])*(1.+xi[1])*(xi[1]-xi[0]-1.)
    N[4] = 0.5*(1.-xi[0]*xi[0])*(1.-xi[1])
    N[5] = 0.5*(1.+xi[0])*(1.-xi[1]*xi[1])
    N[6] = 0.5*(1.-xi[0]*xi[0])*(1.+xi[1])
    N[7] = 0.5*(1.-xi[0])*(1.-xi[1]*xi[1])
    return N

## Shape Functions derivatives

def shapeder(xi):
    
    dNdxi=np.zeros([8,2])
    
    dNdxi[0,0] = 0.25*(1.-xi[1])*(2.*xi[0]+xi[1])
    dNdxi[0,1] = 0.25*(1.-xi[0])*(xi[0]+2.*xi[1])
    dNdxi[1,0] = 0.25*(1.-xi[1])*(2.*xi[0]-xi[1])
    dNdxi[1,1] = 0.25*(1.+xi[0])*(2.*xi[1]-xi[0])
    dNdxi[2,0] = 0.25*(1.+xi[1])*(2.*xi[0]+xi[1])
    dNdxi[2,1] = 0.25*(1.+xi[0])*(2.*xi[1]+xi[0])
    dNdxi[3,0] = 0.25*(1.+xi[1])*(2.*xi[0]-xi[1])
    dNdxi[3,1] = 0.25*(1.-xi[0])*(2.*xi[1]-xi[0])
    dNdxi[4,0] = -xi[0]*(1.-xi[1])
    dNdxi[4,1] = -0.5*(1.-xi[0]*xi[0])
    dNdxi[5,0] = 0.5*(1.-xi[1]*xi[1])
    dNdxi[5,1] = -(1.+xi[0])*xi[1]
    dNdxi[6,0] = -xi[0]*(1.+xi[1])
    dNdxi[6,1] = 0.5*(1.-xi[0]*xi[0])
    dNdxi[7,0] = -0.5*(1.-xi[1]*xi[1])
    dNdxi[7,1] = -(1.-xi[0])*xi[1]
    return dNdxi
    
## This Subrutine calculates the RHS(Residual) in elresid, and K in elstiff

def UEL(coord,materialprops,displacement):
    ncoord=2
    ndof=2
    nelnodes=8
    npoints=9
    dNdx = np.zeros([nelnodes,ncoord])
    dxdxi = np.zeros([ncoord,ncoord])
    STRAIN = np.zeros([9,2,2])
    kel = np.zeros([ndof*nelnodes,ndof*nelnodes])
    rel=np.zeros([nelnodes*ndof,1])
    
    xilist = GSPT()
    w = GPSW()
    xi=np.zeros([2])
    
    STRESS=np.zeros([9,2,2])
    
    ### Loop over integration points
    
    for k in range(npoints):    
        for i in range(ncoord):
            xi[i]=xilist[i,k]
        N=shapefunc(xi)
        dNdxi=shapeder(xi)
        
        dxdxi=np.zeros([ncoord,ncoord])
        
        #Calculation of the Jacobian
        
        for i in range(ncoord):
            for j in range(ncoord):
                dxdxi[i,j]=0.
                for a in range(nelnodes):
                    dxdxi[i,j]=dxdxi[i,j]+coord[i,a]*dNdxi[a,j]
        
        ## INVERSE OF THE JACOBIAN
        dxidx=np.linalg.inv(dxdxi)
        ## DETERMINANT OF THE INVERSE OF THE JACOBIAN
        DJACOB=np.linalg.det(dxdxi)

                       
        ## To calculate the strain gradient, we must calculate the derivatives 
        ## with respect to global coordinates
        
        dNdx=np.zeros([nelnodes,ncoord])
        
        for a in range(nelnodes):
            for i in range(ncoord):
                dNdx[a,i]=0.
                for j in range(ncoord):
                    dNdx[a,i]=dNdx[a,i]+dNdxi[a,j]*dxidx[j,i]
        
                  
                    
        ## The deformation gradients, are calculated differentiating the 
        ## displacements with respect to global coordiantes
        
        F=np.zeros([ncoord,ncoord])
        
        for i in range(ncoord):
            for j in range(ncoord):
                F[i,j]=0
                if (i==j):
                    ## Identity Matrix
                    F[i,i]=1.
                for a in range(nelnodes):
                    F[i,j]=F[i,j]+(displacement[i,a]*dNdx[a,j])
         
                    
        ## J and B are calculated. 
        
        J=np.linalg.det(F)
        B = np.dot(F,np.transpose(F)) ### Strains in this model
        
        ## Computation of shape funtion derivatives for stiffness matrix
        ## dNdyi in the documentation
        
        invF=np.linalg.inv(F)
        
        dNdxs=np.zeros([nelnodes,ncoord])
        
        for a in range(nelnodes):
            for i in range(ncoord):
                dNdxs[a,i]=0.
                for j in range(ncoord):
                    dNdxs[a,i]=dNdxs[a,i]+dNdx[a,j]*invF[j,i]
        
        ### Computation of Kirshooff Stress for Stiffness Matrix
        kstress = Kirchhoffstress(ndof,ncoord,B,J,materialprops)
        
        ## Cauchy Stress
        
        cstress=kstress/J
        STRESS[k,:,:]=cstress
        STRAIN[k,:,:]=B

        
        ## Tha tangent stiffness is calculated, calling the materialstiffness 
        ## Subroutine
        
        C=materialstiffness(ndof,ncoord,B,J,materialprops)
       
        
        ### With all components of the Stiffness Matrix calculated, we must ...
        ### introduce the integration scheme for stifness matrix
        
        for a in range(nelnodes):
            for i in range(ndof):
                for b in range(nelnodes):
                    for m in range(ndof):
                        row=ndof*a+i
                        col=ndof*(b)+m
                        for j in range(ncoord):
                            for l in range(ncoord):
                                kel[row,col]=(kel[row,col]+C[i,j,m,l]*dNdxs[b,l]*
                                              dNdxs[a,j]*w[k]*DJACOB)
                            kel[row,col]=(kel[row,col]-kstress[i,j]*dNdxs[a,m]*dNdxs[b,j]*
                                          w[k]*DJACOB)
        ## Residual Vector Calculation
        
        for a in range(nelnodes):
            for i in range(ndof):
                row=ndof*a+i
                for j in range(ncoord):
                    rel[row]=rel[row]+kstress[i,j]*dNdxs[a,j]*w[k]*DJACOB
                
    return kel,rel,STRESS,STRAIN



        
    
### Next Step Code Assembliers of Residual and Stiffness Matrix

### Assembly of Global Residual R from the local residuals

def assemblyres(nelem,coords,materialprops,topo,u):
    
    ## u[k,j] kth node, jth dof
    ## coords[k,j] kth node, jth dof
    
    ndof=2
    ncoord=2
    nnode=8
    nnodes=coords.shape[0]
    ndofs=ndof*nnodes
    R=np.zeros([ndofs,1])
    
    for i in range(nelem):
        
        ## Coordinates of the current element
        if (nelem==1):
            connect=topo[1:]
        else:
            connect=topo[i,1:]
        
        coord=np.zeros([ncoord,nnode])
        displacement=np.zeros([ncoord,nnode])
        
        for j in range(nnode):
            for k in range(ncoord):
                coord[k,j]=coords[connect[j],k+1]
                displacement[k,j]=u[ndof*connect[j]+k]   
              
        kel,rel,STRESS,STRAIN=UEL(coord,materialprops,displacement)

        
        for a in range(nnode):
            for m in range(ndof):
                R[connect[a]*2+m]=R[connect[a]*2+m]+rel[2*a+m]
                
    return R
        
    ## Stiffness Matrix Assembly
    
def assemblystiff(nelem,coords,materialprops,topo,u):
    
    STRESSG=np.zeros([nelem,2,2])
    
    ## u[k,j] kth node, jth dof
    ## coords[k,j] kth node, jth dof
    
    ndof=2
    ncoord=2
    nnode=8
    nnodes=coords.shape[0]
    ndofs=ndof*nnodes
    
    KG=np.zeros([ndofs,ndofs])

    for i in range(nelem):
        ## Coordinates of the current element
        if (nelem==1):
            connect=topo[1:]
        else:
            connect=topo[i,1:]
        
        coord=np.zeros([ncoord,nnode])
        displacement=np.zeros([ncoord,nnode])
        for j in range(nnode):
            for k in range(ncoord):
                # print(connect[j])
                coord[k,j]=coords[connect[j],k+1]
                displacement[k,j]=u[ndof*connect[j]+k]    
              
        kel,rel,STRESS,STRAIN=UEL(coord,materialprops,displacement)
        STRESSG[i,:,:]=STRESS[4,:,:]
        # print(kel)
        for a in range(nnode):
            for m in range(ndof):
                for b in range(nnode):
                    for n in range(ndof):
                        KG[connect[a]*2+m,connect[b]*2+n]=(KG[connect[a]*2+m,connect[b]*2+n]
                                                  +kel[2*a+m,2*b+n])
                        
    return KG,STRESSG
        
    
def nodalforces(coords,nelem,loads):
    # This subroutine creates the RHS of the system based on the nodal forces introduced
    # in the input file. No distribuited loads included. To do so, the integration scheme 
    # must be implemented.
    
    ndof=2
    ncoord=2
    nnode=8
    nload=loads.shape[0] ## The loads array must be a vector with (1. Node, 2. DOF, 3. Val)
    nnodes=coords.shape[0]
    ndofs=ndof*nnodes
    RHS=np.zeros([ndofs,1])
    
    ### DOF=0 x
    ### DOF=1 y 
    
    for i in range(nload):
        RHS[2*loads[i,0].astype(int)+loads[i,1].astype(int)]=loads[i,2]
    return RHS

def strain_nodes(nelem,coords,materialprops,topo,u):
    """Compute averaged strains and stresses at nodes

    First, the variable is extrapolated from the Gauss
    point to nodes for each element. Then, these values are averaged
    according to the number of element that share that node. The theory
    for this technique can be found in [1]_.

    Parameters
    ----------
    nodes : ndarray (float).
        Array with nodes coordinates.
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    mats : ndarray (float)
        Array with material profiles.
    UC : ndarray (float)
        Array with the displacements. This one contains both, the
        computed and imposed values.

    Returns
    -------
    E_nodes : ndarray
        Strains evaluated at the nodes.


    References
    ----------
    .. [1] O.C. Zienkiewicz and J.Z. Zhu, The Superconvergent patch
        recovery and a posteriori error estimators. Part 1. The
        recovery technique, Int. J. Numer. Methods Eng., 33,
        1331-1364 (1992).
    """
    ndof=2
    ncoord=2
    nnode=8
    nnodes=coords.shape[0]

    E_nodes = np.zeros([nnodes, 3])
    S_nodes = np.zeros([nnodes, 3])
    el_nodes = np.zeros([nnodes], dtype=int)

    IPCON=[0,2,8,6,1,5,7,3]        
            
    for i in range(nelem):
        ## Coordinates of the current element
        if (nelem==1):
            connect=topo[1:]
        else:
            connect=topo[i,1:]
        
        coord=np.zeros([ncoord,nnode])
        displacement=np.zeros([ncoord,nnode])
        for j in range(nnode):
            for k in range(ncoord):
                # print(connect[j])
                coord[k,j]=coords[connect[j],k+1]
                displacement[k,j]=u[ndof*connect[j]+k]    
              
        kel,rel,STRESS,STRAIN=UEL(coord,materialprops,displacement)
        
    

        for cont, node in enumerate(connect):
            E_nodes[node, 0] = E_nodes[node, 0] + STRAIN[IPCON[cont], 0,0]+STRAIN[IPCON[4], 0,0]/8
            E_nodes[node, 1] = E_nodes[node, 1] + STRAIN[IPCON[cont], 1,1]+STRAIN[IPCON[4], 1,1]/8
            E_nodes[node, 2] = E_nodes[node, 2] + STRAIN[IPCON[cont], 0,1]+STRAIN[IPCON[4], 0,1]/8

            S_nodes[node, 0] = S_nodes[node, 0] + STRESS[IPCON[cont], 0,0]+STRESS[IPCON[4], 0,0]/8
            S_nodes[node, 1] = S_nodes[node, 1] + STRESS[IPCON[cont], 1,1]+STRESS[IPCON[4], 1,1]/8
            S_nodes[node, 2] = S_nodes[node, 2] + STRESS[IPCON[cont], 0,1]+STRESS[IPCON[4], 0,1]/8
            el_nodes[node] = el_nodes[node] + 1

    E_nodes[:, 0] = E_nodes[:, 0]/el_nodes
    E_nodes[:, 1] = E_nodes[:, 1]/el_nodes
    E_nodes[:, 2] = E_nodes[:, 2]/el_nodes
    S_nodes[:, 0] = S_nodes[:, 0]/el_nodes
    S_nodes[:, 1] = S_nodes[:, 1]/el_nodes
    S_nodes[:, 2] = S_nodes[:, 2]/el_nodes
    return E_nodes, S_nodes        
    











    
    
