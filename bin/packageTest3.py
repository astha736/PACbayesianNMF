# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 14:31:02 2016

@author: astha

    Copyright (C) 2016   Inria

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description: This module is an implementation of the algorithm presented in 
             P. Alquier and B. Guedj (2016), "A Sharp Oracle Inequality for 
             Bayesian Non-Negative Matrix Factorization" (arXiv preprint).
             

Requirement: Script has been tested on 
             "PACBayesianNMF"   v0.1.0
             "numpy"            v1.11.0
             "matplotlib"       v1.5.0

Purpose:    This script is written to load data for digits as given by 
            train.txt. Each row is a digit represented by label as first
            column followed by flattened 16x16 image represented as a array
            of 256 elements.
            
            Each element is preprocessed to belong to a number between [0,1]
            Then the conditions for block gradient descent are set and algorithm
            is applied to generate U, V, crit and out as output.
            U  and V   :are factors of the original dataMatrix.
            crit       :is an array of distance of estimated matrix
                        UV from original dataMatrix 
            out        :list of values of exit condition of the three loops.
                        This can be used for debugging purposes 
            
    See script in bin for sample usage of the package 
"""
from pacbayesiannmf import *
import numpy as np 
import matplotlib.pyplot as plt

################## Loading Data into a matrix ################################
print "************  opening file train.txt **************"
try:
    file = open ( 'train.txt' , 'r')
except:
    print "Error: Not able to open file"
    
#dataList = [ map(float,line.split()) for line in file ]

dataList = []
temp = []
for line in file:
    for i in line.split(): 
        temp.append(float(i))
    dataList.append(temp)
    temp = []


file.close()

dataList = np.array(dataList)
   
###############################################################################
########################## Pre-processing #####################################
########################### Initialization ####################################
# selection is used to select data points for digit < selection 
selection = 2
# cleaning for the digits we need in the data matrix 
dataList = dataList[dataList[:,0] < selection]

dataMatrix = np.matrix(dataList) 

# size of the dataMatrix m1 and m2 
m1, m2 = dataMatrix.shape

# seperating labels and data points 
labels = dataMatrix[:,0]
dataMatrix = dataMatrix[:,1:m2]

m2 = m2-1

dataMatrix = np.matrix(dataMatrix)
# to bring each element in between 0 to 1 
dataMatrix = np.add(dataMatrix,1)/2
###############################################################################
##################### Call blockGradintDescent################################
shp =  dataMatrix.shape
if len(shp) != 2:
    print "Error: Please change the script for image to have a 2d matrix"
    print "       Current Shape of Matrix is:"+str(shp)
elif np.min(dataMatrix) < 0 or np.max(dataMatrix) > 1:
    print "Error: Values not between 0 and 1"
    print "       Please modify script to adjust values"
else:
    print"*******Call to blockGradientDescent******"
    z = blockGradientDescent(dataMatrix,2)
    z.setConditionOnAllSteps(3e-1,1e-3,1e-3)
    (U,V,crit,out)= z.applyBlockGradientDescent(printflag = 1)
    print"*******End of blockGradientDescent*******"
    ########################## Plot #######################################
    V = V*255
    #U = U*255
    V[V < 0] = 0
    V[V > 255] = 255
    #U[U < 0] = 0
    #U[U > 255] = 255
    V.astype(int)
    
    f, axarr = plt.subplots(2,3)
    axarr[0,0].imshow(np.reshape(V[:,0],(16,16)),cmap='Greys_r')
    axarr[0,1].imshow(np.reshape(V[:,1],(16,16)),cmap='Greys_r')
    axarr[1,2].plot(crit)
    ############################# END #########################################
#################### End of the script ########################################