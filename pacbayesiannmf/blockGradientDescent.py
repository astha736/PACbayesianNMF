# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:34:04 2016

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

    See script in bin for sample usage of the package 
"""

import numpy as np 

class blockGradientDescent:
    def __init__(self,dataMatrix,K):
        #set all the initialization 
        # Matrix that contains data 
        # Each row is a data point with each point between 0 and 1         
        self.dataMatrix = dataMatrix
        # K defines the number of signals to separate out from the data matrix 
        self.K = K
        # condition on each step for stopping block descent 
        # this says that the difference of each step should be > condition_on_* vale  
        # condition_on_step for the outer most loop 
        self.condition_on_step = 1e-2 
        # condition_on_inside_step_U for steps on U 
        self.condition_on_inside_step_U = 1e-3 
        # condition_on_inside_step_V for steps on V 
        self.condition_on_inside_step_V = 1e-3 
    
    def setDataMatrix(self,dataMatrix):
        self.dataMatrix = dataMatrix
    
    def setNoOfPatterns(self,K):
        self.K = K
    
    def setConditionOnAllSteps(self,condition_on_step = 1e-2,condition_on_inside_step_U = 1e-3,condition_on_inside_step_V = 1e-3):
        self.condition_on_step = condition_on_step #5e-7
        self.condition_on_inside_step_U = condition_on_inside_step_U #1e-8
        self.condition_on_inside_step_V = condition_on_inside_step_V #1e-8
    
    def setConditionOnOutsideStep(self,condition_on_step = 1e-2):
        self.condition_on_step = condition_on_step #5e-7
    
    def setConditionOnInsideStepU(self,condition_on_inside_step_U = 1e-3):
        self.condition_on_inside_step_U = condition_on_inside_step_U #1e-8
    
    def setConditionOnInsideStepV(self,condition_on_inside_step_V = 1e-3):
        self.condition_on_inside_step_V = condition_on_inside_step_V #1e-8
        
    def applyBlockGradientDescent(self,b = 1e6,lmbd = (float(1)/4)*100,pas = 1e-3,printflag = 0): 
        print "******** Start: Apply Block Gradient Descent *****************"
        # size of the dataMatrix m1 and m2 
        m1, m2 = self.dataMatrix.shape
        
        # enforces sparsity 
        b = b
        #lambda serves as constant to initialize lmbd2 
        lmbd = float(lmbd)
        # serves as a constant to initialize another constant pas2 
        pas = pas
        
        # out is a list containing various values for debugging purposes 
        out = []
        # initialization
        # matrix U and V, according to the algorithm 
        # random value generator for exponential distribution for rate 
        # here rate is set to 1 
        # np.matrix(np.reshape(np.random.exponential(1,m1*K),(m1,K)))
        # matrix(create 2d array(from a list of numbers generated from exponential dist))
        U = np.matrix(np.reshape(np.random.exponential(1,m1*self.K),(m1,self.K)))
        V = np.matrix(np.reshape(np.random.exponential(1,m2*self.K),(m2,self.K)))
        # matrix multiplication of U and transpose(V) represented as UV  
        UV = np.dot(U,np.matrix.transpose(V))
        # creates an array of length K and initializes it with int 1 
        gamma = np.matrix(np.ones(self.K))
        
        # keeps count of the outer loop 
        k=0 
        # set the initial distance to Inf or has high value as possible 
        eps = float('Inf')

        pas2 = 2*pas
        lmbd2 = 2*lmbd

        # can say to be distance function 
        # a integer value is assigned after calculation 
        crit = []
        crit.append(np.sum(np.power((self.dataMatrix-UV),2)))
        
        # mixed algorithm: block coordinate descent with projected gradient
        # eps is set Inf in the starting 
        
        while eps > self.condition_on_step: 
            k = k+1
            ####### variables that are required to be reset for each loop ######
            # 1 is for the first while loop (U) and 2 is for the second (V) 
            # assignment to new variable 
            uv = UV
            # assignment to new variable
            eps_inside1 = float('Inf')
            # assignment to new variable
            k_inside1 = 1
            # assignment to new variable
            eps_inside2 = float('Inf')
            # assignment to new variable
            k_inside2 = 1
            ###################################################################
            # keep repeating till the eps_inside1> self.condition_on_inside_step_U (which is the distance)
            while eps_inside1> self.condition_on_inside_step_U:
                u = U
                # block descent step 
                U = U + pas2/(np.sqrt(k_inside1))*(np.dot((self.dataMatrix-np.dot(U,np.matrix.transpose(V))),V) - np.repeat(gamma,m1,axis=0)/lmbd2)
                # remove any negative entries, projection step
                # since our assumption is that U and V are also non-negative 
                U[U<0]=0
                # update criteria for moving out of the loop
                eps_inside1 = np.sum(np.power((u-U),2))
                # k_inside1 the number of iterations for U 
                k_inside1 = k_inside1 + 1
                
        	# keep repeating till the eps_inside2> self.condition_on_inside_step_V (which is the distance)
        	while eps_inside2> self.condition_on_inside_step_V:
                 v = V
                 # block descent step 
                 V = V + pas2/(np.sqrt(k_inside2))*(np.dot((np.matrix.transpose(self.dataMatrix) - np.dot(V,np.matrix.transpose(U))),U) - np.repeat(gamma,m2,axis=0)/lmbd2)
                 # remove any negative entries, projection step 
                 # since our assumption is that U and V are also non-negative 
                 V[V<0]=0
                 # update criteria for moving out of the loop 
                 eps_inside2 = np.sum(np.power((v-V),2)) 
                 # k_inside2 the number of iterations for V 
                 k_inside2 = k_inside2 + 1
        
            # UV calculated based on the new values from the descent 
            UV = np.dot(U,np.matrix.transpose(V))
            # calculate sum of the columns 
            # vector of length = K 
            colsums = np.sum(U,axis=0)+np.sum(V,axis =0)
            # use this colsums to calculate gamma 
            # vector of length = K 
            # element wise multiplication and other operators 
            try: 
                gamma = np.multiply(np.sqrt(colsums/b),(np.sqrt(1+9/(16*b*colsums))-3/(4*np.sqrt(b*colsums))))
            except:
                gamma = np.matrix(np.zeros(self.K))
            # distance from the previous UV 
            # update criteria for moving out of the loop 
            eps = np.sum(np.power((uv-UV),2))
            # appending the distance from actual array to be approximated (X) in crit
            crit.append(np.sum(np.power((self.dataMatrix-UV),2)))
            
            outline = "Iteration: "+str(k)+ "\t"+"Iteration for U: "+str(k_inside1)+"\t"+ "Iteration for V:"+str(k_inside2)+"\t"+"eps: "
            outline = outline + str(eps) + '\n'
            if printflag == 1:
                print outline
            out.append(outline) 
        print "********** End: Apply Block Gradient Descent *****************"
        return (U,V,crit,out)