
import torch
torch.manual_seed(0)

import random
# random.seed(0)

import numpy as np
# np.random.seed(0)

import torch.optim as optim

import time

import os

from matplotlib import pyplot as plt

from sklearn.utils import shuffle

from NN import Net, on_task_update, train_ewc, test


from scripts import mnist, n_mnist, fashion_mnist, oversample
from scripts.instances_generator import dset, New_instances_suite_1, New_instances_suite_4
from scripts.logs import logs

from hyperopt import fmin, tpe, hp





# ======================================================================
# == Check GPU is connected
# ======================================================================

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");


# ======================================================================
# == Initial training datasets prep
# ======================================================================
mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

# ======================================================================
# == Retraining datasets prep
# ======================================================================
fashion_mnist.init()

X_train_fashion, Y_train_fashion, X_test_fashion, Y_test_fashion, fashion_dataset = fashion_mnist.load()

# gn_X_train, gn_Y_train ,gn_X_test, gn_Y_test, gn_dataset                              = n_mnist.load("gn") # load gaussian nosie dataset
# blur_X_train, blur_Y_train ,blur_X_test, blur_Y_test, blur_dataset                    = n_mnist.load("blur") # load blur nosie dataset
# contrast_X_train, contrast_Y_train ,contrast_X_test,contrast_Y_test, contrast_dataset = n_mnist.load("contrast") # load contrast noise dataset

# DONE TODO extract the test dataset for each digit from each noise 
digit_0_test_X = X_test[np.where(Y_test==0)]
digit_1_test_X = X_test[np.where(Y_test==1)]
digit_2_test_X = X_test[np.where(Y_test==2)]
digit_3_test_X = X_test[np.where(Y_test==3)]
digit_4_test_X = X_test[np.where(Y_test==4)]
digit_5_test_X = X_test[np.where(Y_test==5)]
digit_6_test_X = X_test[np.where(Y_test==6)]
digit_7_test_X = X_test[np.where(Y_test==7)]
digit_8_test_X = X_test[np.where(Y_test==8)]
digit_9_test_X = X_test[np.where(Y_test==9)]

digit_0_test_Y = Y_test[np.where(Y_test==0)]
digit_1_test_Y = Y_test[np.where(Y_test==1)]
digit_2_test_Y = Y_test[np.where(Y_test==2)]
digit_3_test_Y = Y_test[np.where(Y_test==3)]
digit_4_test_Y = Y_test[np.where(Y_test==4)]
digit_5_test_Y = Y_test[np.where(Y_test==5)]
digit_6_test_Y = Y_test[np.where(Y_test==6)]
digit_7_test_Y = Y_test[np.where(Y_test==7)]
digit_8_test_Y = Y_test[np.where(Y_test==8)]
digit_9_test_Y = Y_test[np.where(Y_test==9)]


fashion_0_test_X = X_test_fashion[np.where(Y_test_fashion==0)]
fashion_1_test_X = X_test_fashion[np.where(Y_test_fashion==1)]
fashion_2_test_X = X_test_fashion[np.where(Y_test_fashion==2)]
fashion_3_test_X = X_test_fashion[np.where(Y_test_fashion==3)]
fashion_4_test_X = X_test_fashion[np.where(Y_test_fashion==4)]
fashion_5_test_X = X_test_fashion[np.where(Y_test_fashion==5)]
fashion_6_test_X = X_test_fashion[np.where(Y_test_fashion==6)]
fashion_7_test_X = X_test_fashion[np.where(Y_test_fashion==7)]
fashion_8_test_X = X_test_fashion[np.where(Y_test_fashion==8)]
fashion_9_test_X = X_test_fashion[np.where(Y_test_fashion==9)]

fashion_0_test_Y = Y_test_fashion[np.where(Y_test_fashion==0)]
fashion_1_test_Y = Y_test_fashion[np.where(Y_test_fashion==1)]
fashion_2_test_Y = Y_test_fashion[np.where(Y_test_fashion==2)]
fashion_3_test_Y = Y_test_fashion[np.where(Y_test_fashion==3)]
fashion_4_test_Y = Y_test_fashion[np.where(Y_test_fashion==4)]
fashion_5_test_Y = Y_test_fashion[np.where(Y_test_fashion==5)]
fashion_6_test_Y = Y_test_fashion[np.where(Y_test_fashion==6)]
fashion_7_test_Y = Y_test_fashion[np.where(Y_test_fashion==7)]
fashion_8_test_Y = Y_test_fashion[np.where(Y_test_fashion==8)]
fashion_9_test_Y = Y_test_fashion[np.where(Y_test_fashion==9)]

# for img in fashion_0_test_X:
# 	plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
# 	plt.savefig('batch.png')
# 	input("press enter")



# retraining sets representing the open world environment
n_retraining_sets  = 10#30  # Number of differnt retraining sets
num_of_oversamples = 1000 # number of oversampled instances to be generated
experiments_number = 1#20

max_rotation = 30
max_shift    = 0.2



# ======================================================================
# == Initialising logging arrays
# ======================================================================
# Initialising first logging set arrays  
log_experiment_ID_array                                                          = []
log_current_ID_array                                                             = []
log_added_instance_digit_array                                                   = []
log_added_instance_noise_type_array                                              = []
log_initial_dataset_size_array                                                   = []
log_oversampling_size_array                                                      = []
log_accumilated_retraining_dataset_size_array                                    = []
log_lr_retraining_array                                                          = []
log_ewc_lambda_array                                                             = []

log_first_model_acc_on_original_training_data                                    = []
log_first_model_acc_on_original_testing_data                                     = []
log_first_model_acc_on_oversampled_new_instance_training_data                    = []
log_first_model_acc_on_oversampled_new_instance_testing_data                     = []
log_first_model_acc_on_accumilated_oversampled_new_instances_training_data       = []
log_first_model_acc_on_accumilated_oversampled_new_instances_testing_data        = []

log_before_last_model_acc_on_original_training_data                              = []
log_before_last_model_acc_on_original_testing_data                               = []
log_before_last_model_acc_on_oversampled_new_instance_training_data              = []
log_before_last_model_acc_on_oversampled_new_instance_testing_data               = []
log_before_last_model_acc_on_accumilated_oversampled_new_instances_training_data = []
log_before_last_model_acc_on_accumilated_oversampled_new_instances_testing_data  = []

log_last_model_acc_on_original_training_data                                     = []
log_last_model_acc_on_original_testing_data                                      = []
log_last_model_acc_on_oversampled_new_instance_training_data                     = []
log_last_model_acc_on_oversampled_new_instance_testing_data                      = []
log_last_model_acc_on_accumilated_oversampled_new_instances_training_data        = []
log_last_model_acc_on_accumilated_oversampled_new_instances_testing_data         = []

log_acc_instance_test_first_model = []
log_acc_instance_test_before_last_model = []
log_acc_instance_test_last_model = []

log_acc_accumilated_instances_test_first_model = []
log_acc_accumilated_instances_test_before_last_model = []
log_acc_accumilated_instances_test_last_model = []


# Initialising second logging set arrays  
# DONE TODO create logging arrays

log_acc_digit_0_original_first_model       =  []
log_acc_digit_1_original_first_model       =  []
log_acc_digit_2_original_first_model       =  []
log_acc_digit_3_original_first_model       =  []
log_acc_digit_4_original_first_model       =  []
log_acc_digit_5_original_first_model       =  []
log_acc_digit_6_original_first_model       =  []
log_acc_digit_7_original_first_model       =  []
log_acc_digit_8_original_first_model       =  []
log_acc_digit_9_original_first_model       =  []

log_acc_digit_0_original_last_model       =  []
log_acc_digit_1_original_last_model       =  []
log_acc_digit_2_original_last_model       =  []
log_acc_digit_3_original_last_model       =  []
log_acc_digit_4_original_last_model       =  []
log_acc_digit_5_original_last_model       =  []
log_acc_digit_6_original_last_model       =  []
log_acc_digit_7_original_last_model       =  []
log_acc_digit_8_original_last_model       =  []
log_acc_digit_9_original_last_model       =  []

log_acc_digit_0_noise_0_first_model       =  []
log_acc_digit_1_noise_0_first_model       =  []
log_acc_digit_2_noise_0_first_model       =  []
log_acc_digit_3_noise_0_first_model       =  []
log_acc_digit_4_noise_0_first_model       =  []
log_acc_digit_5_noise_0_first_model       =  []
log_acc_digit_6_noise_0_first_model       =  []
log_acc_digit_7_noise_0_first_model       =  []
log_acc_digit_8_noise_0_first_model       =  []
log_acc_digit_9_noise_0_first_model       =  []

log_acc_digit_0_noise_0_last_model       =  []
log_acc_digit_1_noise_0_last_model       =  []
log_acc_digit_2_noise_0_last_model       =  []
log_acc_digit_3_noise_0_last_model       =  []
log_acc_digit_4_noise_0_last_model       =  []
log_acc_digit_5_noise_0_last_model       =  []
log_acc_digit_6_noise_0_last_model       =  []
log_acc_digit_7_noise_0_last_model       =  []
log_acc_digit_8_noise_0_last_model       =  []
log_acc_digit_9_noise_0_last_model       =  []

log_digit_0_noise_0_flag   =  []
log_digit_1_noise_0_flag   =  []
log_digit_2_noise_0_flag   =  []
log_digit_3_noise_0_flag   =  []
log_digit_4_noise_0_flag   =  []
log_digit_5_noise_0_flag   =  []
log_digit_6_noise_0_flag   =  []
log_digit_7_noise_0_flag   =  []
log_digit_8_noise_0_flag   =  []
log_digit_9_noise_0_flag   =  []


log_acc_all_digits_original_first_model = []
log_acc_all_digits_original_last_model  = []
log_acc_all_digits_noise_0_first_model  = []
log_acc_all_digits_noise_0_last_model   = []


# Initialising third logging set arrays
log_experiment_ID_array2 = []
log_newest_instance_ID   = []
log_instance_ID          = []
log_instance_digit       = []
log_instance_noise_type  = []
log_acc_instance_before_last_model = []
log_acc_instance_last_model = []
log_acc_on_task_first_model = []
log_acc_on_task_before_last_model = []
log_acc_on_task_last_model = []



# ======================================================================
# == Define retraining objective function
# ======================================================================
def objective(hyper_params):#ewc_lambda_hyper, lr_value_retrain_hyper):#(ewc_lambda, lr_value_retrain): 
	# def objective(ewc_lambda): 
	print(hyper_params["x"])
	print(hyper_params["x"][0])
	print(hyper_params["x"][1])
	ewc_lambda_hyper = hyper_params["x"][0]
	lr_value_retrain_hyper = hyper_params["x"][1]
	# lr_value_retrain_hyper = 0.01
	print(ewc_lambda_hyper)

	ewc_lambdas_hyper = [ewc_lambda_hyper]

	#retrain
	#################### WARNING ewc and lr numbers will not be updated in the print below as they are being done internally by Hyperopt.
	print("==============================================================")
	print("Experiment ID = ",experiment_ID,", Retraining ID = ",current_ID,", oversamples_num = ",oversamples_num,", lr_value_retrain = ",lr_value_retrain_hyper,", ewc_lambda = ",ewc_lambda_hyper)
	print("==============================================================")

	global retrained_model
	retrained_model = Net().to(device)
	retrained_model.load_state_dict(trained_models[0].state_dict()) # Use initial model to do the retraining instead of the last retrainied model
	optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_retrain_hyper, momentum=momentum_value_retrain)


	for epoch in range(0, epoch_retrain):
		retrained_model,_,_ = train_ewc(retrained_model, device, 1, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y, optimizer, epoch, ewc_lambdas_hyper, fisher_dict, optpar_dict)
		# TODO Add if out puting Nan break for loop

	# Get the score of Original tasks on the retrained model: get performance on each tasks individiually, subtract from theshold, get the max diff.
	array_of_original_tasks_scores = [] 
	for i in range(10):
		temp_diff = original_tasks_acc_threshold - test(retrained_model, device, X_test[np.where(Y_test==i)], Y_test[np.where(Y_test==i)])
		if temp_diff > 0:
			array_of_original_tasks_scores.append(temp_diff)
		else:
			array_of_original_tasks_scores.append(0)

		print("Original task", i," performance = ", test(retrained_model, device, X_test[np.where(Y_test==i)], Y_test[np.where(Y_test==i)]))
		print(" theshold - performance = ", original_tasks_acc_threshold - test(retrained_model, device, X_test[np.where(Y_test==i)], Y_test[np.where(Y_test==i)]))
		print("===")
	# original_tasks_score   = np.max(array_of_original_tasks_scores)
	original_tasks_score   = sum(array_of_original_tasks_scores)

	# Get the score of PoL tasks on the retrained model
	accumilted_tasks_score = 100 - test(retrained_model, device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)

	# Overall score
	# if accumilted_tasks_score == 0:
	# 	retraining_score = original_tasks_score + 10000000
	# else:
	# 	retraining_score = original_tasks_score + 1/accumilted_tasks_score

	
	retraining_score = original_tasks_score + accumilted_tasks_score


	# if len(array_of_original_tasks_scores) > 0:
	# 	retraining_score = original_tasks_score/accumilted_tasks_score

	# elif len(array_of_original_tasks_scores) == 0:
	# 	retraining_score = 1/accumilted_tasks_score

	temp_retrained_models_array.append(retrained_model)
	temp_retraining_scores_array.append(retraining_score)
	# retraining_counter += 1
	# print("retraining counter = ", retraining_counter)
	print("original_tasks_score = " ,original_tasks_score)
	print("accumilted_tasks_score = " ,accumilted_tasks_score)
	print("retraining_score = " ,retraining_score)
	print("=============================================")
	# input("Press enter to continue")

	return retraining_score




for experiment_ID in range(experiments_number):

	# ======================================================================
	# == Retraining datasets generation
	# ======================================================================
	# TODO: Edit fucntion below to choose which tasks to generate instances from
	retraining_sets = New_instances_suite_1(n_retraining_sets*5, num_of_oversamples, \
									max_rotation,max_shift, \
									fashion_dataset)

	# retraining_sets = New_instances_suite_4(n_retraining_sets*4, num_of_oversamples, \
	# 								max_rotation,max_shift, \
	# 								[gn_digit_9_train_X, gn_digit_9_train_Y, gn_digit_9_test_X, gn_digit_9_test_Y],\
	# 								[blur_digit_2_train_X,blur_digit_2_train_Y, blur_digit_2_test_X,blur_digit_2_test_Y],\
	# 								[blur_digit_3_train_X,blur_digit_3_train_Y, blur_digit_3_test_X,blur_digit_3_test_Y],\
	# 								[blur_digit_5_train_X,blur_digit_5_train_Y,blur_digit_5_test_X,blur_digit_5_test_Y],\
	# 								[blur_digit_7_train_X,blur_digit_7_train_Y, blur_digit_7_test_X,blur_digit_7_test_Y],\
	# 								[blur_digit_8_train_X,blur_digit_8_train_Y,blur_digit_8_test_X,blur_digit_8_test_Y],\
	# 								[blur_digit_9_train_X,blur_digit_9_train_Y,blur_digit_9_test_X,blur_digit_9_test_Y],\
	# 								[contrast_digit_2_train_X,contrast_digit_2_train_X,contrast_digit_2_test_X,contrast_digit_2_test_X],\
	# 								[contrast_digit_3_train_X,contrast_digit_3_train_X,contrast_digit_3_test_X,contrast_digit_3_test_X],\
	# 								[contrast_digit_5_train_X,contrast_digit_5_train_X,contrast_digit_5_test_X,contrast_digit_5_test_X],\
	# 								[contrast_digit_7_train_X,contrast_digit_7_train_X,contrast_digit_7_test_X,contrast_digit_7_test_X],\
	# 								[contrast_digit_8_train_X,contrast_digit_8_train_X,contrast_digit_8_test_X,contrast_digit_8_test_X],\
	# 								[contrast_digit_9_train_X,contrast_digit_9_train_X,contrast_digit_9_test_X,contrast_digit_9_test_X]
	# 								)
	################## WARNING if I change the list of tasks inputed above, change accoridngly the noise labeling 

	# retraining_sets = New_instances_suite_2(gn_dataset, blur_dataset, contrast_dataset)

	print("len(retraining_sets) = ", len(retraining_sets))


	# ======================================================================
	# == Declarations
	# ======================================================================
	Misclassification_threshold         = 50 # Set the threshold for retraining.
	Misclassification_counter           = 0  # Counter to count number of mis-classifications.
	Misclassification_counter_threshold = 5 # Max number of misclassifications allowed to do retraining for.

	Initial_training_flag = True

	Retraining_flag       = True

	Accumilation_flag     = True  


	fisher_dict = {}
	optpar_dict = {}

	results_logs = logs()

	# ======================================================================
	# == Initial Training
	# ======================================================================

	if Initial_training_flag == True:
		Initial_training_flag = False

		init_start_time = time.time()
		
		lr_value_init       = 0.01   # learning rate for initial learning
		momentum_value_init = 0.9    # momentum value for initial learning
		epoch_init          = 30     # epoch value
		ewc_lambda          = 0      # setting ewc_lambda to 0, so no retraining factor  


		model = Net().to(device)
		optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

		for epoch in range(0, epoch_init):
			model, fisher_dict, optpar_dict = train_ewc(model, device, 0, X_train, Y_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
		fisher_dict, optpar_dict = on_task_update(0, X_train, Y_train, model, optimizer, fisher_dict, optpar_dict)

		init_end_time1 = time.time()

	# ======================================================================
	# == Evaluate Initial Training model
	# ======================================================================

	acc_train_init_orig = test(model, device, X_train, Y_train) # Evaluate model on initial training data
	print("Accuracy on initial training data (X_train, Y_train)", acc_train_init_orig)

	acc_test_init_orig = test(model, device, X_test, Y_test)   # Evaluate model on initial testing data
	print("Accuracy on initial testing data (X_test, Y_test)", acc_test_init_orig)

	init_end_time2 = time.time()
	init_training_cost = init_end_time1 - init_start_time
	print("Cost of initial training = ",init_training_cost)
	init_training_evaluation_cost = init_end_time2 - init_start_time
	print("Cost of initial training = ",init_training_evaluation_cost)

	

	# ======================================================================
	# == Sequential Retraining 
	# ======================================================================
	lr_value_retrain       = 0.01   # learning rate for retraining learning
	momentum_value_retrain = 0.9    # momentum value for retraining learning
	epoch_retrain          = 30     # epoch value 
	ewc_lambdas            = []
	trained_models         = [model] 

	retrained_sets = []
	# retrained_sets_counter = 0
	hypertuning_flag = True
	oversampling_range = [1000]#[50, 100, 500, 1000] # try 1000 as well
	ewc_lambda_range   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #[0,10,50,100] #[0, 1, 5, 10, 100] 
	lr_learning_range  = [0.0001,0.0002, 0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,\
							0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01] #[0.0001, 0.0002, 0.0005, 0.0007, 0.01] 



	digit_0_noise_0_counter = 0
	digit_1_noise_0_counter = 0
	digit_2_noise_0_counter = 0
	digit_3_noise_0_counter = 0
	digit_4_noise_0_counter = 0
	digit_5_noise_0_counter = 0
	digit_6_noise_0_counter = 0
	digit_7_noise_0_counter = 0
	digit_8_noise_0_counter = 0
	digit_9_noise_0_counter = 0




	i = 0 
	if Retraining_flag == True :

		for current_ID in range(n_retraining_sets):
			
			# Find a new instance from the set of generated retraining instances
			while True:
				# select retraining set
				retraining_set = retraining_sets[i]
				print("retraining_set.train_X.dtype",retraining_set.train_X.dtype) 
				print("retraining_set.train_X.shape",retraining_set.train_X.shape) 

				# for img in retraining_set.train_X:
				# 	plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
				# 	plt.savefig('batch.png')
				# 	input("press enter")




				#TODO Move oversampling code to here and assess on the oversamples. This might affect the oversamples control in training...?

				# assess its accuracy on last trained model
				acc = test(trained_models[-1], device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
				# print("trained_models[-1] = ", trained_models[-1])
				# print("device",device)
				# print("retraining_set.train_X.dtype = ", retraining_set.train_X.dtype)
				# print("retraining_set.train_X.shape = ", retraining_set.train_X.shape)

				# print("retraining_set.train_Y.dtype = ", retraining_set.train_Y.dtype)
				# print("retraining_set.train_Y.shape = ", retraining_set.train_Y.shape)
				# input("press enter")

				# if retraining_set.noise_type == 0 and retraining_set.label == 9:
				# 	retrainable_instance_flag = True

				# elif retraining_set.noise_type == 1 and retraining_set.label == 2:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 1 and retraining_set.label == 3:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 1 and retraining_set.label == 5:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 1 and retraining_set.label == 7:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 1 and retraining_set.label == 8:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 1 and retraining_set.label == 9:
				# 	retrainable_instance_flag = True

				# elif retraining_set.noise_type == 2 and retraining_set.label == 2:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 2 and retraining_set.label == 3:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 2 and retraining_set.label == 5:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 2 and retraining_set.label == 7:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 2 and retraining_set.label == 8:
				# 	retrainable_instance_flag = True
				# elif retraining_set.noise_type == 2 and retraining_set.label == 9:
				# 	retrainable_instance_flag = True

				# else:
				# 	retrainable_instance_flag = False

				# if accuracy below threshold then:
				if acc < Misclassification_threshold:# and retrainable_instance_flag:

					# i)   set the retrainingset ID to the current ID
					retraining_set.ID = current_ID

					# ii)  append the retarining set to the set of retrained sets
					retrained_sets.append(retraining_set)

					# iii) break while loop
					i += 1
					break

				else:
					i += 1
					print("#########################################")
					print("Retraining set is above the threshold")
					print("#########################################")

				print("new instances set investigated i = ",i,"/",len(retraining_sets))




			# Loop over the different oversampling rates and run experiments with them
			temp_models_array        = []
			temp_ovesamples_array    = []
			temp_lr_retraining_array = []
			temp_ewc_array           = []

			start_time = time.time()

			for oversamples_num in oversampling_range:
				# DONE TODO select the oversamples from the 1000 samples to generate the accumilated dataset
				# Create the accumilated retraining set 
				temp_i = 0
				for retraining_set in retrained_sets:
					if temp_i == 0:
						train_X_temp = retraining_set.train_X[:oversamples_num]
						train_Y_temp = retraining_set.train_Y[:oversamples_num]
						test_X_temp = retraining_set.test_X
						test_Y_temp = retraining_set.test_Y
					else:
						train_X_temp = np.concatenate((train_X_temp, retraining_set.train_X[:oversamples_num]), axis=0)
						train_Y_temp = np.concatenate((train_Y_temp, retraining_set.train_Y[:oversamples_num]), axis=0)
						test_X_temp = np.concatenate((test_X_temp, retraining_set.test_X), axis=0)
						test_Y_temp = np.concatenate((test_Y_temp, retraining_set.test_Y), axis=0)

					temp_i += 1
				# Shuffle
				train_X_temp, train_Y_temp = shuffle(train_X_temp, train_Y_temp)

				noise_type_ID = 10   # mix of noise datasets
				dataset_ID = 1000000 # mix of noise datasets
				test_X_all_temp = None
				test_Y_all_temp = None
				accumilated_retraining_set = dset(dataset_ID, noise_type_ID,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all_temp, test_Y_all_temp)
# ====================
				# TODO tuning implementation move here from line 610
				
				# Assess if retrained_model fits the criteria below:
				original_tasks_acc_threshold = 90

				if 'fisher_dict' in globals():
					print("Yes fisher is in it")
				else:
					print("No fisher is not in it")

				# Minimise this retraining score: minimize the objective over the space.
				temp_retrained_models_array = []
				temp_retraining_scores_array = []
				retraining_counter = 0
				print("=============================================")
				print("=============================================")
				# objfunc = lambda x: objective(x)
				best = fmin(fn=lambda x: objective(x),
						# space={
						# 	hp.uniform('ewc_lambda_hyper', 0, 100), 
						# 	hp.uniform('lr_value_retrain_hyper', 0.0001,0.01)
						# 	},
						space= {'x': [hp.uniform('ewc_lambda_hyper', 0, 100), hp.uniform('lr_value_retrain_hyper', 0.0001,0.01)]},
						 	# {
							# 'ewc_lambda_hyper': hp.uniform('ewc_lambda_hyper', 0, 100), 
							# 'lr_value_retrain_hyper': hp.uniform('lr_value_retrain_hyper', 0.0001,0.01)
							# },
						# space=hp.choice('retraining_parameters', [
						# 	{
						# 	'ewc_lambda_hyper': hp.uniform('ewc_lambda_hyper', 0, 100), 
						# 	'lr_value_retrain_hyper': hp.uniform('lr_value_retrain_hyper', 0.0001,0.01)
						# 	}
						# 	]),
						algo=tpe.suggest,
						max_evals=100)
			

				print("Best hyper parameters = ",best)
				print("best ewc_lambda_hyper = ",best["ewc_lambda_hyper"])
				print("best lr_value_retrain_hyper   = ",best["lr_value_retrain_hyper"])


				temp_retrained_models_array = np.array(temp_retrained_models_array)
				temp_retraining_scores_array = np.array(temp_retraining_scores_array)

				# Get the index of the best score i.e. minimum
				best_score = min(temp_retraining_scores_array)
				best_score_index = np.where(temp_retraining_scores_array == best_score)

				# Use the index found to select the best 
				retrained_model = temp_retrained_models_array[best_score_index[0][0]]
			

				# STOPPED HERE: 
				# I need to  remove the steps below.
				# Append model from tuning to the trained arrays.
				# Run
				# The retrained model is from the objective function, need to check if it is the same model being used here or not.

# =====================
			
			# Based on the scoring choose which model to choose as the retrained model
			# retrained_model = temp_models_array[index_of_best_scoring_model]

			# Append to trained models array the model giving the best score. 
			trained_models.append(retrained_model)

			# Update counters for instances of each new task
			if retraining_set.noise_type == 0 and retraining_set.label == 0:
				digit_0_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 1:
				digit_1_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 2:
				digit_2_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 3:
				digit_3_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 4:
				digit_4_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 5:
				digit_5_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 6:
				digit_6_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 7:
				digit_7_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 8:
				digit_8_noise_0_counter += 1
			elif retraining_set.noise_type == 0 and retraining_set.label == 9:
				digit_9_noise_0_counter += 1



			# Creating accumilated tasks dataset based on the digit and noise types of the new instances involved in the accumilated retrainng dataset. 

			array_of_tasks_X       = np.array([fashion_0_test_X,fashion_1_test_X,fashion_2_test_X,fashion_3_test_X,fashion_4_test_X,fashion_5_test_X,fashion_6_test_X,fashion_7_test_X,fashion_8_test_X,fashion_9_test_X])

			array_of_tasks_Y       = np.array([fashion_0_test_Y,fashion_1_test_Y,fashion_2_test_Y,fashion_3_test_Y,fashion_4_test_Y,fashion_5_test_Y,fashion_6_test_Y,fashion_7_test_Y,fashion_8_test_Y,fashion_9_test_Y])

			array_of_tasks_flags = np.array([digit_0_noise_0_counter,digit_1_noise_0_counter,digit_2_noise_0_counter,digit_3_noise_0_counter,digit_4_noise_0_counter,digit_5_noise_0_counter,digit_6_noise_0_counter,digit_7_noise_0_counter,digit_8_noise_0_counter,digit_9_noise_0_counter])
			
			temp_tasks  = array_of_tasks_X[np.where(array_of_tasks_flags > 0)]
			temp_labels = array_of_tasks_Y[np.where(array_of_tasks_flags > 0)]

			accumilated_new_tasks_involved_from_new_instances_test_dataset_X = []
			accumilated_new_tasks_involved_from_new_instances_test_dataset_Y = []

			for temp_ii in range(len(temp_labels)):
				if temp_ii == 0:
					accumilated_new_tasks_involved_from_new_instances_test_dataset_X = array_of_tasks_X[temp_ii]
					accumilated_new_tasks_involved_from_new_instances_test_dataset_Y = array_of_tasks_Y[temp_ii]
			
				else:
					accumilated_new_tasks_involved_from_new_instances_test_dataset_X = np.concatenate((accumilated_new_tasks_involved_from_new_instances_test_dataset_X, array_of_tasks_X[temp_ii]), axis=0)
					accumilated_new_tasks_involved_from_new_instances_test_dataset_Y = np.concatenate((accumilated_new_tasks_involved_from_new_instances_test_dataset_Y, array_of_tasks_Y[temp_ii]), axis=0)


			
			# DONE TODO populate the first logging arrays
			log_experiment_ID_array.append(experiment_ID)
			log_current_ID_array.append(current_ID) 
			log_added_instance_noise_type_array.append(retraining_set.noise_type)
			log_initial_dataset_size_array.append(60000)
			log_oversampling_size_array.append(oversampling_range[0])#temp_ovesamples_array[index_of_best_scoring_model])
			log_accumilated_retraining_dataset_size_array.append(len(accumilated_retraining_set.train_Y))
			log_lr_retraining_array.append(best["lr_value_retrain_hyper"])#temp_lr_retraining_array[index_of_best_scoring_model])
			log_ewc_lambda_array.append(best["ewc_lambda_hyper"])#temp_ewc_array[index_of_best_scoring_model])

			log_first_model_acc_on_original_training_data.append(test(trained_models[0], device, X_train, Y_train)) 
			log_first_model_acc_on_original_testing_data.append(test(trained_models[0], device, X_test, Y_test))
			log_first_model_acc_on_oversampled_new_instance_training_data.append(test(trained_models[0], device, retraining_set.train_X, retraining_set.train_Y))
			log_first_model_acc_on_oversampled_new_instance_testing_data.append(test(trained_models[0], device, retraining_set.test_X, retraining_set.test_Y))
			log_first_model_acc_on_accumilated_oversampled_new_instances_training_data.append(test(trained_models[0], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y))
			log_first_model_acc_on_accumilated_oversampled_new_instances_testing_data.append(test(trained_models[0], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y))

			log_before_last_model_acc_on_original_training_data.append(test(trained_models[-2], device, X_train, Y_train))
			log_before_last_model_acc_on_original_testing_data.append(test(trained_models[-2], device, X_test, Y_test))
			log_before_last_model_acc_on_oversampled_new_instance_training_data.append(test(trained_models[-2], device, retraining_set.train_X, retraining_set.train_Y))
			log_before_last_model_acc_on_oversampled_new_instance_testing_data.append(test(trained_models[-2], device, retraining_set.test_X, retraining_set.test_Y))
			log_before_last_model_acc_on_accumilated_oversampled_new_instances_training_data.append(test(trained_models[-2], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y))
			log_before_last_model_acc_on_accumilated_oversampled_new_instances_testing_data.append(test(trained_models[-2], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y))

			log_last_model_acc_on_original_training_data.append(test(trained_models[-1], device, X_train, Y_train))
			log_last_model_acc_on_original_testing_data.append(test(trained_models[-1], device, X_test, Y_test))
			log_last_model_acc_on_oversampled_new_instance_training_data.append(test(trained_models[-1], device, retraining_set.train_X, retraining_set.train_Y))
			log_last_model_acc_on_oversampled_new_instance_testing_data.append(test(trained_models[-1], device, retraining_set.test_X, retraining_set.test_Y))
			log_last_model_acc_on_accumilated_oversampled_new_instances_training_data.append(test(trained_models[-1], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y))
			log_last_model_acc_on_accumilated_oversampled_new_instances_testing_data.append(test(trained_models[-1], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y))
			##
			log_acc_instance_test_first_model.append(test(trained_models[0], device, retraining_set.test_X, retraining_set.test_Y)) 
			log_acc_instance_test_before_last_model.append(test(trained_models[-2], device, retraining_set.test_X, retraining_set.test_Y))
			log_acc_instance_test_last_model.append(test(trained_models[-1], device, retraining_set.test_X, retraining_set.test_Y))

			log_acc_accumilated_instances_test_first_model.append(test(trained_models[0], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)) 
			log_acc_accumilated_instances_test_before_last_model.append(test(trained_models[-2], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)) 
			log_acc_accumilated_instances_test_last_model.append(test(trained_models[-1], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)) 


			# TODO populate the second set logging arrays
			# populat the digita array too, do not forget
			log_added_instance_digit_array.append(retraining_set.label)


			log_acc_digit_0_original_first_model.append(test(trained_models[0], device, digit_0_test_X, digit_0_test_Y))
			log_acc_digit_1_original_first_model.append(test(trained_models[0], device, digit_1_test_X, digit_1_test_Y))
			log_acc_digit_2_original_first_model.append(test(trained_models[0], device, digit_2_test_X, digit_2_test_Y))
			log_acc_digit_3_original_first_model.append(test(trained_models[0], device, digit_3_test_X, digit_3_test_Y))
			log_acc_digit_4_original_first_model.append(test(trained_models[0], device, digit_4_test_X, digit_4_test_Y))
			log_acc_digit_5_original_first_model.append(test(trained_models[0], device, digit_5_test_X, digit_5_test_Y))
			log_acc_digit_6_original_first_model.append(test(trained_models[0], device, digit_6_test_X, digit_6_test_Y))
			log_acc_digit_7_original_first_model.append(test(trained_models[0], device, digit_7_test_X, digit_7_test_Y))
			log_acc_digit_8_original_first_model.append(test(trained_models[0], device, digit_8_test_X, digit_8_test_Y))
			log_acc_digit_9_original_first_model.append(test(trained_models[0], device, digit_9_test_X, digit_9_test_Y))

			log_acc_digit_0_original_last_model.append(test(trained_models[-1], device, digit_0_test_X, digit_0_test_Y))
			log_acc_digit_1_original_last_model.append(test(trained_models[-1], device, digit_1_test_X, digit_1_test_Y))
			log_acc_digit_2_original_last_model.append(test(trained_models[-1], device, digit_2_test_X, digit_2_test_Y))
			log_acc_digit_3_original_last_model.append(test(trained_models[-1], device, digit_3_test_X, digit_3_test_Y))
			log_acc_digit_4_original_last_model.append(test(trained_models[-1], device, digit_4_test_X, digit_4_test_Y))
			log_acc_digit_5_original_last_model.append(test(trained_models[-1], device, digit_5_test_X, digit_5_test_Y))
			log_acc_digit_6_original_last_model.append(test(trained_models[-1], device, digit_6_test_X, digit_6_test_Y))
			log_acc_digit_7_original_last_model.append(test(trained_models[-1], device, digit_7_test_X, digit_7_test_Y))
			log_acc_digit_8_original_last_model.append(test(trained_models[-1], device, digit_8_test_X, digit_8_test_Y))
			log_acc_digit_9_original_last_model.append(test(trained_models[-1], device, digit_9_test_X, digit_9_test_Y))

			log_acc_digit_0_noise_0_first_model.append(test(trained_models[0], device, fashion_0_test_X, fashion_0_test_Y))
			log_acc_digit_1_noise_0_first_model.append(test(trained_models[0], device, fashion_1_test_X, fashion_1_test_Y))
			log_acc_digit_2_noise_0_first_model.append(test(trained_models[0], device, fashion_2_test_X, fashion_2_test_Y))
			log_acc_digit_3_noise_0_first_model.append(test(trained_models[0], device, fashion_3_test_X, fashion_3_test_Y))
			log_acc_digit_4_noise_0_first_model.append(test(trained_models[0], device, fashion_4_test_X, fashion_4_test_Y))
			log_acc_digit_5_noise_0_first_model.append(test(trained_models[0], device, fashion_5_test_X, fashion_5_test_Y))
			log_acc_digit_6_noise_0_first_model.append(test(trained_models[0], device, fashion_6_test_X, fashion_6_test_Y))
			log_acc_digit_7_noise_0_first_model.append(test(trained_models[0], device, fashion_7_test_X, fashion_7_test_Y))
			log_acc_digit_8_noise_0_first_model.append(test(trained_models[0], device, fashion_8_test_X, fashion_8_test_Y))
			log_acc_digit_9_noise_0_first_model.append(test(trained_models[0], device, fashion_9_test_X, fashion_9_test_Y))

			log_acc_digit_0_noise_0_last_model.append(test(trained_models[-1], device, fashion_0_test_X, fashion_0_test_Y))
			log_acc_digit_1_noise_0_last_model.append(test(trained_models[-1], device, fashion_1_test_X, fashion_1_test_Y))
			log_acc_digit_2_noise_0_last_model.append(test(trained_models[-1], device, fashion_2_test_X, fashion_2_test_Y))
			log_acc_digit_3_noise_0_last_model.append(test(trained_models[-1], device, fashion_3_test_X, fashion_3_test_Y))
			log_acc_digit_4_noise_0_last_model.append(test(trained_models[-1], device, fashion_4_test_X, fashion_4_test_Y))
			log_acc_digit_5_noise_0_last_model.append(test(trained_models[-1], device, fashion_5_test_X, fashion_5_test_Y))
			log_acc_digit_6_noise_0_last_model.append(test(trained_models[-1], device, fashion_6_test_X, fashion_6_test_Y))
			log_acc_digit_7_noise_0_last_model.append(test(trained_models[-1], device, fashion_7_test_X, fashion_7_test_Y))
			log_acc_digit_8_noise_0_last_model.append(test(trained_models[-1], device, fashion_8_test_X, fashion_8_test_Y))
			log_acc_digit_9_noise_0_last_model.append(test(trained_models[-1], device, fashion_9_test_X, fashion_9_test_Y))

			log_digit_0_noise_0_flag.append(digit_0_noise_0_counter)
			log_digit_1_noise_0_flag.append(digit_1_noise_0_counter)
			log_digit_2_noise_0_flag.append(digit_2_noise_0_counter)
			log_digit_3_noise_0_flag.append(digit_3_noise_0_counter)
			log_digit_4_noise_0_flag.append(digit_4_noise_0_counter)
			log_digit_5_noise_0_flag.append(digit_5_noise_0_counter)
			log_digit_6_noise_0_flag.append(digit_6_noise_0_counter)
			log_digit_7_noise_0_flag.append(digit_7_noise_0_counter)
			log_digit_8_noise_0_flag.append(digit_8_noise_0_counter)
			log_digit_9_noise_0_flag.append(digit_9_noise_0_counter)

	

			log_acc_all_digits_original_first_model.append(test(trained_models[0], device, X_test, Y_test))
			log_acc_all_digits_original_last_model.append(test(trained_models[-1], device, X_test, Y_test))
			log_acc_all_digits_noise_0_first_model.append(test(trained_models[0], device, X_test_fashion, Y_test_fashion))
			log_acc_all_digits_noise_0_last_model.append(test(trained_models[-1], device, X_test_fashion, Y_test_fashion))



			# Populate the third set of logs
			for re_set in retrained_sets:
				log_experiment_ID_array2.append(experiment_ID)
				log_newest_instance_ID.append(current_ID)
				log_instance_ID.append(re_set.ID)
				log_instance_digit.append(re_set.label)
				log_instance_noise_type.append(re_set.noise_type)
				log_acc_instance_before_last_model.append(test(trained_models[-2], device, re_set.test_X, re_set.test_Y))
				log_acc_instance_last_model.append(test(trained_models[-1], device, re_set.test_X, re_set.test_Y))
				
				if re_set.noise_type == 0 and re_set.label == 0:
					temp_task_test_X = fashion_0_test_X
					temp_task_test_Y = fashion_0_test_Y
				elif re_set.noise_type == 0 and re_set.label == 1:
					temp_task_test_X = fashion_1_test_X
					temp_task_test_Y = fashion_1_test_Y
				elif re_set.noise_type == 0 and re_set.label == 2:
					temp_task_test_X = fashion_2_test_X
					temp_task_test_Y = fashion_2_test_Y
				elif re_set.noise_type == 0 and re_set.label == 3:
					temp_task_test_X = fashion_3_test_X
					temp_task_test_Y = fashion_3_test_Y
				elif re_set.noise_type == 0 and re_set.label == 4:
					temp_task_test_X = fashion_4_test_X
					temp_task_test_Y = fashion_4_test_Y
				elif re_set.noise_type == 0 and re_set.label == 5:
					temp_task_test_X = fashion_5_test_X
					temp_task_test_Y = fashion_5_test_Y
				elif re_set.noise_type == 0 and re_set.label == 6:
					temp_task_test_X = fashion_6_test_X
					temp_task_test_Y = fashion_6_test_Y
				elif re_set.noise_type == 0 and re_set.label == 7:
					temp_task_test_X = fashion_7_test_X
					temp_task_test_Y = fashion_7_test_Y
				elif re_set.noise_type == 0 and re_set.label == 8:
					temp_task_test_X = fashion_8_test_X
					temp_task_test_Y = fashion_8_test_Y
				elif re_set.noise_type == 0 and re_set.label == 9:
					temp_task_test_X = fashion_9_test_X
					temp_task_test_Y = fashion_9_test_Y

				

				log_acc_on_task_first_model.append(test(trained_models[0], device, temp_task_test_X, temp_task_test_Y))
				log_acc_on_task_before_last_model.append(test(trained_models[-2], device, temp_task_test_X, temp_task_test_Y))
				log_acc_on_task_last_model.append(test(trained_models[-1], device, temp_task_test_X, temp_task_test_Y))
				

# Write log file

# Folder "results" if not already there
output_folder = "Results_logs"
file_name1     = "results_combined_34-1.csv"
file_name2     = "results_combined_34-2.csv"
file_name3     = "results_combined_34-3.csv"

if not os.path.exists(output_folder):
	os.makedirs(output_folder)

# Write first set of logs
file_path = os.path.join(output_folder, file_name1)
with open(file_path, 'w') as log_file: 
	log_file.write('experiment_ID, ID ,noise type, Initial training dataset size, Oversampling Size per new instance, accumilated_retraining_dataset_size, lr, ewc_lambda, \
					first_model_acc_on_original_training_data, first_model_acc_on_original_testing_data,\
					first_model_acc_on_oversampled_new_instance_training_data, first_model_acc_on_oversampled_new_instance_testing_data,\
					first_model_acc_on_accumilated_oversampled_new_instances_training_data, first_model_acc_on_accumilated_oversampled_new_instances_testing_data,\
					before_last_model_acc_on_original_training_data, before_last_model_acc_on_original_testing_data,\
					before_last_model_acc_on_oversampled_new_instance_training_data, before_last_model_acc_on_oversampled_new_instance_testing_data,\
					before_last_model_acc_on_accumilated_oversampled_new_instances_training_data, before_last_model_acc_on_accumilated_oversampled_new_instances_testing_data,\
					last_model_acc_on_original_training_data, last_model_acc_on_original_testing_data,\
					last_model_acc_on_oversampled_new_instance_training_data, last_model_acc_on_oversampled_new_instance_testing_data,\
					last_model_acc_on_accumilated_oversampled_new_instances_training_data, last_model_acc_on_accumilated_oversampled_new_instances_testing_data,\
					acc_instance_test_first_model, acc_instance_test_before_last_model, acc_instance_test_last_model,\
					acc_accumilated_instances_test_first_model, acc_accumilated_instances_test_before_last_model, acc_accumilated_instances_test_last_model\n')

	for i in range(len(log_current_ID_array)):
		log_file.write('%d,%d,%d,%d,%d,%d,%1.6f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n' %\
					(log_experiment_ID_array[i], log_current_ID_array[i], log_added_instance_noise_type_array[i],\
					log_initial_dataset_size_array[i],log_oversampling_size_array[i],\
					log_accumilated_retraining_dataset_size_array[i], log_lr_retraining_array[i],log_ewc_lambda_array[i],\
					log_first_model_acc_on_original_training_data[i],log_first_model_acc_on_original_testing_data[i],\
					log_first_model_acc_on_oversampled_new_instance_training_data[i],log_first_model_acc_on_oversampled_new_instance_testing_data[i],\
					log_first_model_acc_on_accumilated_oversampled_new_instances_training_data[i],log_first_model_acc_on_accumilated_oversampled_new_instances_testing_data[i],\
					log_before_last_model_acc_on_original_training_data[i],log_before_last_model_acc_on_original_testing_data[i],\
					log_before_last_model_acc_on_oversampled_new_instance_training_data[i],log_before_last_model_acc_on_oversampled_new_instance_testing_data[i],
					log_before_last_model_acc_on_accumilated_oversampled_new_instances_training_data[i],log_before_last_model_acc_on_accumilated_oversampled_new_instances_testing_data[i],
					log_last_model_acc_on_original_training_data[i],log_last_model_acc_on_original_testing_data[i],\
					log_last_model_acc_on_oversampled_new_instance_training_data[i],log_last_model_acc_on_oversampled_new_instance_testing_data[i],\
					log_last_model_acc_on_accumilated_oversampled_new_instances_training_data[i],log_last_model_acc_on_accumilated_oversampled_new_instances_testing_data[i],\
					log_acc_instance_test_first_model[i],\
					log_acc_instance_test_before_last_model[i],\
					log_acc_instance_test_last_model[i],\
					log_acc_accumilated_instances_test_first_model[i],\
					log_acc_accumilated_instances_test_before_last_model[i],\
					log_acc_accumilated_instances_test_last_model[i]))

# DONE TODO Write second set of logs
file_path = os.path.join(output_folder, file_name2)
with open(file_path, 'w') as log_file: 
	log_file.write('experiment_ID, \
					ID, \
					latest_new_instance_digit, \
					noise type, \
					Initial training dataset size, \
					Oversampling Size per new instance, \
					accumilated_retraining_dataset_size, \
					lr, \
					ewc_lambda, \
					\
					acc_digit_0_original_first_model,\
					acc_digit_1_original_first_model,\
					acc_digit_2_original_first_model,\
					acc_digit_3_original_first_model,\
					acc_digit_4_original_first_model,\
					acc_digit_5_original_first_model,\
					acc_digit_6_original_first_model,\
					acc_digit_7_original_first_model,\
					acc_digit_8_original_first_model,\
					acc_digit_9_original_first_model,\
\
					acc_digit_0_original_last_model,\
					acc_digit_1_original_last_model,\
					acc_digit_2_original_last_model,\
					acc_digit_3_original_last_model,\
					acc_digit_4_original_last_model,\
					acc_digit_5_original_last_model,\
					acc_digit_6_original_last_model,\
					acc_digit_7_original_last_model,\
					acc_digit_8_original_last_model,\
					acc_digit_9_original_last_model,\
\
					acc_digit_0_noise_0_first_model,\
					acc_digit_1_noise_0_first_model,\
					acc_digit_2_noise_0_first_model,\
					acc_digit_3_noise_0_first_model,\
					acc_digit_4_noise_0_first_model,\
					acc_digit_5_noise_0_first_model,\
					acc_digit_6_noise_0_first_model,\
					acc_digit_7_noise_0_first_model,\
					acc_digit_8_noise_0_first_model,\
					acc_digit_9_noise_0_first_model,\
\
					acc_digit_0_noise_0_last_model,\
					acc_digit_1_noise_0_last_model,\
					acc_digit_2_noise_0_last_model,\
					acc_digit_3_noise_0_last_model,\
					acc_digit_4_noise_0_last_model,\
					acc_digit_5_noise_0_last_model,\
					acc_digit_6_noise_0_last_model,\
					acc_digit_7_noise_0_last_model,\
					acc_digit_8_noise_0_last_model,\
					acc_digit_9_noise_0_last_model,\
\
					acc_digit_0_noise_0_flag,\
					acc_digit_1_noise_0_flag,\
					acc_digit_2_noise_0_flag,\
					acc_digit_3_noise_0_flag,\
					acc_digit_4_noise_0_flag,\
					acc_digit_5_noise_0_flag,\
					acc_digit_6_noise_0_flag,\
					acc_digit_7_noise_0_flag,\
					acc_digit_8_noise_0_flag,\
					acc_digit_9_noise_0_flag,\
\
					acc_all_digits_original_first_model,\
					acc_all_digits_original_last_model,\
					acc_all_digits_noise_0_first_model,\
					acc_all_digits_noise_0_last_model,\
					\n')

	for i in range(len(log_current_ID_array)):
		log_file.write('%d,%d,%d,%d,\
						%d,%d,%d,%1.6f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f\n' %\
					(log_experiment_ID_array[i], log_current_ID_array[i], log_added_instance_digit_array[i], log_added_instance_noise_type_array[i],\
					log_initial_dataset_size_array[i],log_oversampling_size_array[i], log_accumilated_retraining_dataset_size_array[i], log_lr_retraining_array[i],log_ewc_lambda_array[i],\
					log_acc_digit_0_original_first_model[i],\
					log_acc_digit_1_original_first_model[i],\
					log_acc_digit_2_original_first_model[i],\
					log_acc_digit_3_original_first_model[i],\
					log_acc_digit_4_original_first_model[i],\
					log_acc_digit_5_original_first_model[i],\
					log_acc_digit_6_original_first_model[i],\
					log_acc_digit_7_original_first_model[i],\
					log_acc_digit_8_original_first_model[i],\
					log_acc_digit_9_original_first_model[i],\

					log_acc_digit_0_original_last_model[i],\
					log_acc_digit_1_original_last_model[i],\
					log_acc_digit_2_original_last_model[i],\
					log_acc_digit_3_original_last_model[i],\
					log_acc_digit_4_original_last_model[i],\
					log_acc_digit_5_original_last_model[i],\
					log_acc_digit_6_original_last_model[i],\
					log_acc_digit_7_original_last_model[i],\
					log_acc_digit_8_original_last_model[i],\
					log_acc_digit_9_original_last_model[i],\

					log_acc_digit_0_noise_0_first_model[i],\
					log_acc_digit_1_noise_0_first_model[i],\
					log_acc_digit_2_noise_0_first_model[i],\
					log_acc_digit_3_noise_0_first_model[i],\
					log_acc_digit_4_noise_0_first_model[i],\
					log_acc_digit_5_noise_0_first_model[i],\
					log_acc_digit_6_noise_0_first_model[i],\
					log_acc_digit_7_noise_0_first_model[i],\
					log_acc_digit_8_noise_0_first_model[i],\
					log_acc_digit_9_noise_0_first_model[i],\

					log_acc_digit_0_noise_0_last_model[i],\
					log_acc_digit_1_noise_0_last_model[i],\
					log_acc_digit_2_noise_0_last_model[i],\
					log_acc_digit_3_noise_0_last_model[i],\
					log_acc_digit_4_noise_0_last_model[i],\
					log_acc_digit_5_noise_0_last_model[i],\
					log_acc_digit_6_noise_0_last_model[i],\
					log_acc_digit_7_noise_0_last_model[i],\
					log_acc_digit_8_noise_0_last_model[i],\
					log_acc_digit_9_noise_0_last_model[i],\

					log_digit_0_noise_0_flag[i],\
					log_digit_1_noise_0_flag[i],\
					log_digit_2_noise_0_flag[i],\
					log_digit_3_noise_0_flag[i],\
					log_digit_4_noise_0_flag[i],\
					log_digit_5_noise_0_flag[i],\
					log_digit_6_noise_0_flag[i],\
					log_digit_7_noise_0_flag[i],\
					log_digit_8_noise_0_flag[i],\
					log_digit_9_noise_0_flag[i],\

					log_acc_all_digits_original_first_model[i],\
					log_acc_all_digits_original_last_model[i],\
					log_acc_all_digits_noise_0_first_model[i],\
					log_acc_all_digits_noise_0_last_model[i]))


file_path = os.path.join(output_folder, file_name3)
with open(file_path, 'w') as log_file: 
	log_file.write('experiment_ID, \
					ID, \
					instance_ID, \
					instance_digit,\
					instance_noise_type, \
					acc_instance_before_last_model, \
					acc_instance_last_model,\
					acc_on_task_first_model,\
					acc_on_task_before_last_model, \
					acc_on_task_last_model\
					\n')

	for i in range(len(log_experiment_ID_array2)):
		log_file.write('%d,%d,%d,%d,%d,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n' %\
					(log_experiment_ID_array2[i], log_newest_instance_ID[i], log_instance_ID[i], log_instance_digit[i], log_instance_noise_type[i], log_acc_instance_before_last_model[i], log_acc_instance_last_model[i], log_acc_on_task_first_model[i], log_acc_on_task_before_last_model[i], log_acc_on_task_last_model[i]))


print('Log files SUCCESSFULLY generated!')
