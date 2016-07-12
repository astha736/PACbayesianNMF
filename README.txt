################################# README.txt###############################################
# Name of the project	: PACBayesianNMF 

# Purpose 				: Implementation of a PAC-Bayesian approach to Non-Negative Matrix

#						  Factorization, usein block gradient descent.

# Authors				: Astha Gupta 		<astha736@gmail.com>

#						  Benjamin Guedj 	<benjamin.guedj@inria.fr>

# Reference 			: Source: https://arxiv.org/abs/1601.01345

#						  P. Alquier and B. Guedj (2016). "A Sharp Oracle Inequality

#						  for Bayesian Non-Negative Matrix Factorization", arXiv preprint.

#						  Please refer to the paper above for a full description on the 

#						  implemented algorithm.

#License				: GPLv3 

#########################################################################################

# List of important files and functions 

# 		#################################################################################

#	  	".\pacbayesiannmf\blockGradientDescent.py"

#	  	Contains class called blockGradientDescent

#		-> setDataMatrix(self,dataMatrix):	function to set dataMatrix 

#		-> setNoOfPatterns(self,K):			function to set no of patterns to find 

#		-> setConditionOnAllSteps(self,concondition_on_step = 1e-2,

#		   condition_on_inside_step_U = 1e-3,

#		   condition_on_inside_step_V = 1e-3): Set exit conditions for block

#											   gradient descent 

#		-> setConditionOnOutsideStep(self,concondition_on_step = 1e-2):

#		   Set value for the most outside step, minimize for UV

#		-> setConditionOnInsideStepU(self,condition_on_inside_step_U = 1e-3):

#		   Set value for the inner loop that minimizes for U

#		-> setConditionOnInsideStepV(self,condition_on_inside_step_V = 1e-3):

#		   Set value for the inner loop that minimized for V

#

#		-> def applyBlockGradientDescent(self,b = 1e6,lmbd = (float(1)/4)*100,

#		   pas = 1e-3,printflag = 0): 

#		   This is the main function that applies blockGradientDescent 

#		   b is used to inforce sparcity 

#		   lmbd is lambda from the algorithm 

#          pas is constant used in algorithm for calculating new U and V 

#		##################################################################################	

##########################################################################################		   

# Usage: Import package, create a 2d dataMatrix with each row as a datapoint having values

# 		 between 0 to 1. Create an object of class with parameters as dataMatrix and an 

#		 integer that specifies no of patterns to be detected. Set other parameters using

#		 set*() methods. Finally to apply Block Gradient Descent use 		

# 		 applyBlockGradientDescent() method 

#

# from pacbayesiannmf import *

# z = blockGradientDescent(dataMatrix,2)

# z.setConditionOnAllSteps(1e-4,1e-6,1e-6)

# (U,V,crit,out)= z.applyBlockGradientDescent(printflag = 1)

#

# Most important output is V, which contains the signal. Each column of V 

# crit gives as an array with distance between actual datamatrix and estimated UV with each step

# out is a list of values of different variables with each step, helps in debugging 

###########################################################################################

################################## END#####################################################

