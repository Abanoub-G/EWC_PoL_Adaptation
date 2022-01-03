
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


from scripts import mnist, n_mnist, oversample
from scripts.instances_generator import dset, New_instances_suite_1, New_instances_suite_2
from scripts.logs import logs





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
n_mnist.init()

gn_X_train, gn_Y_train ,gn_X_test, gn_Y_test, gn_dataset                              = n_mnist.load("gn") # load gaussian nosie dataset
blur_X_train, blur_Y_train ,blur_X_test, blur_Y_test, blur_dataset                    = n_mnist.load("blur") # load blur nosie dataset
contrast_X_train, contrast_Y_train ,contrast_X_test,contrast_Y_test, contrast_dataset = n_mnist.load("contrast") # load contrast noise dataset

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

# for img in digit_1_test:
# 	plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
# 	plt.savefig('batch.png')
# 	input("press enter")

gn_digit_0_test_X = gn_X_test[np.where(gn_Y_test==0)]
gn_digit_1_test_X = gn_X_test[np.where(gn_Y_test==1)]
gn_digit_2_test_X = gn_X_test[np.where(gn_Y_test==2)]
gn_digit_3_test_X = gn_X_test[np.where(gn_Y_test==3)]
gn_digit_4_test_X = gn_X_test[np.where(gn_Y_test==4)]
gn_digit_5_test_X = gn_X_test[np.where(gn_Y_test==5)]
gn_digit_6_test_X = gn_X_test[np.where(gn_Y_test==6)]
gn_digit_7_test_X = gn_X_test[np.where(gn_Y_test==7)]
gn_digit_8_test_X = gn_X_test[np.where(gn_Y_test==8)]
gn_digit_9_test_X = gn_X_test[np.where(gn_Y_test==9)]

gn_digit_0_test_Y = gn_Y_test[np.where(gn_Y_test==0)]
gn_digit_1_test_Y = gn_Y_test[np.where(gn_Y_test==1)]
gn_digit_2_test_Y = gn_Y_test[np.where(gn_Y_test==2)]
gn_digit_3_test_Y = gn_Y_test[np.where(gn_Y_test==3)]
gn_digit_4_test_Y = gn_Y_test[np.where(gn_Y_test==4)]
gn_digit_5_test_Y = gn_Y_test[np.where(gn_Y_test==5)]
gn_digit_6_test_Y = gn_Y_test[np.where(gn_Y_test==6)]
gn_digit_7_test_Y = gn_Y_test[np.where(gn_Y_test==7)]
gn_digit_8_test_Y = gn_Y_test[np.where(gn_Y_test==8)]
gn_digit_9_test_Y = gn_Y_test[np.where(gn_Y_test==9)]


blur_digit_0_test_X = blur_X_test[np.where(blur_Y_test==0)]
blur_digit_1_test_X = blur_X_test[np.where(blur_Y_test==1)]
blur_digit_2_test_X = blur_X_test[np.where(blur_Y_test==2)]
blur_digit_3_test_X = blur_X_test[np.where(blur_Y_test==3)]
blur_digit_4_test_X = blur_X_test[np.where(blur_Y_test==4)]
blur_digit_5_test_X = blur_X_test[np.where(blur_Y_test==5)]
blur_digit_6_test_X = blur_X_test[np.where(blur_Y_test==6)]
blur_digit_7_test_X = blur_X_test[np.where(blur_Y_test==7)]
blur_digit_8_test_X = blur_X_test[np.where(blur_Y_test==8)]
blur_digit_9_test_X = blur_X_test[np.where(blur_Y_test==9)]

blur_digit_0_test_Y = blur_Y_test[np.where(blur_Y_test==0)]
blur_digit_1_test_Y = blur_Y_test[np.where(blur_Y_test==1)]
blur_digit_2_test_Y = blur_Y_test[np.where(blur_Y_test==2)]
blur_digit_3_test_Y = blur_Y_test[np.where(blur_Y_test==3)]
blur_digit_4_test_Y = blur_Y_test[np.where(blur_Y_test==4)]
blur_digit_5_test_Y = blur_Y_test[np.where(blur_Y_test==5)]
blur_digit_6_test_Y = blur_Y_test[np.where(blur_Y_test==6)]
blur_digit_7_test_Y = blur_Y_test[np.where(blur_Y_test==7)]
blur_digit_8_test_Y = blur_Y_test[np.where(blur_Y_test==8)]
blur_digit_9_test_Y = blur_Y_test[np.where(blur_Y_test==9)]


contrast_digit_0_test_X = contrast_X_test[np.where(contrast_Y_test==0)]
contrast_digit_1_test_X = contrast_X_test[np.where(contrast_Y_test==1)]
contrast_digit_2_test_X = contrast_X_test[np.where(contrast_Y_test==2)]
contrast_digit_3_test_X = contrast_X_test[np.where(contrast_Y_test==3)]
contrast_digit_4_test_X = contrast_X_test[np.where(contrast_Y_test==4)]
contrast_digit_5_test_X = contrast_X_test[np.where(contrast_Y_test==5)]
contrast_digit_6_test_X = contrast_X_test[np.where(contrast_Y_test==6)]
contrast_digit_7_test_X = contrast_X_test[np.where(contrast_Y_test==7)]
contrast_digit_8_test_X = contrast_X_test[np.where(contrast_Y_test==8)]
contrast_digit_9_test_X = contrast_X_test[np.where(contrast_Y_test==9)]


contrast_digit_0_test_Y = contrast_Y_test[np.where(contrast_Y_test==0)]
contrast_digit_1_test_Y = contrast_Y_test[np.where(contrast_Y_test==1)]
contrast_digit_2_test_Y = contrast_Y_test[np.where(contrast_Y_test==2)]
contrast_digit_3_test_Y = contrast_Y_test[np.where(contrast_Y_test==3)]
contrast_digit_4_test_Y = contrast_Y_test[np.where(contrast_Y_test==4)]
contrast_digit_5_test_Y = contrast_Y_test[np.where(contrast_Y_test==5)]
contrast_digit_6_test_Y = contrast_Y_test[np.where(contrast_Y_test==6)]
contrast_digit_7_test_Y = contrast_Y_test[np.where(contrast_Y_test==7)]
contrast_digit_8_test_Y = contrast_Y_test[np.where(contrast_Y_test==8)]
contrast_digit_9_test_Y = contrast_Y_test[np.where(contrast_Y_test==9)]

# retraining sets representing the open world environment
n_retraining_sets  = 10#30  # Number of differnt retraining sets
num_of_oversamples = 1000 # number of oversampled instances to be generated
experiments_number = 1

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

# Initialising second logging set arrays  
# DONE TODO create logging arrays

log_acc_digit_0_original       =  []
log_acc_digit_1_original       =  []
log_acc_digit_2_original       =  []
log_acc_digit_3_original       =  []
log_acc_digit_4_original       =  []
log_acc_digit_5_original       =  []
log_acc_digit_6_original       =  []
log_acc_digit_7_original       =  []
log_acc_digit_8_original       =  []
log_acc_digit_9_original       =  []

log_acc_digit_0_noise_0        =  []
log_acc_digit_0_noise_0_flag   =  []
log_acc_digit_1_noise_0        =  []
log_acc_digit_1_noise_0_flag   =  []
log_acc_digit_2_noise_0        =  []
log_acc_digit_2_noise_0_flag   =  []
log_acc_digit_3_noise_0        =  []
log_acc_digit_3_noise_0_flag   =  []
log_acc_digit_4_noise_0        =  []
log_acc_digit_4_noise_0_flag   =  []
log_acc_digit_5_noise_0        =  []
log_acc_digit_5_noise_0_flag   =  []
log_acc_digit_6_noise_0        =  []
log_acc_digit_6_noise_0_flag   =  []
log_acc_digit_7_noise_0        =  []
log_acc_digit_7_noise_0_flag   =  []
log_acc_digit_8_noise_0        =  []
log_acc_digit_8_noise_0_flag   =  []
log_acc_digit_9_noise_0        =  []
log_acc_digit_9_noise_0_flag   =  []

log_acc_digit_0_noise_1        =  []
log_acc_digit_0_noise_1_flag   =  []
log_acc_digit_1_noise_1        =  []
log_acc_digit_1_noise_1_flag   =  []
log_acc_digit_2_noise_1        =  []
log_acc_digit_2_noise_1_flag   =  []
log_acc_digit_3_noise_1        =  []
log_acc_digit_3_noise_1_flag   =  []
log_acc_digit_4_noise_1        =  []
log_acc_digit_4_noise_1_flag   =  []
log_acc_digit_5_noise_1        =  []
log_acc_digit_5_noise_1_flag   =  []
log_acc_digit_6_noise_1        =  []
log_acc_digit_6_noise_1_flag   =  []
log_acc_digit_7_noise_1        =  []
log_acc_digit_7_noise_1_flag   =  []
log_acc_digit_8_noise_1        =  []
log_acc_digit_8_noise_1_flag   =  []
log_acc_digit_9_noise_1        =  []
log_acc_digit_9_noise_1_flag   =  []

log_acc_digit_0_noise_2        =  []
log_acc_digit_0_noise_2_flag   =  []
log_acc_digit_1_noise_2        =  []
log_acc_digit_1_noise_2_flag   =  []
log_acc_digit_2_noise_2        =  []
log_acc_digit_2_noise_2_flag   =  []
log_acc_digit_3_noise_2        =  []
log_acc_digit_3_noise_2_flag   =  []
log_acc_digit_4_noise_2        =  []
log_acc_digit_4_noise_2_flag   =  []
log_acc_digit_5_noise_2        =  []
log_acc_digit_5_noise_2_flag   =  []
log_acc_digit_6_noise_2        =  []
log_acc_digit_6_noise_2_flag   =  []
log_acc_digit_7_noise_2        =  []
log_acc_digit_7_noise_2_flag   =  []
log_acc_digit_8_noise_2        =  []
log_acc_digit_8_noise_2_flag   =  []
log_acc_digit_9_noise_2        =  []
log_acc_digit_9_noise_2_flag   =  []

log_acc_overall_old_tasks_original_model = []
log_acc_overall_old_tasks_before_last_model = []
log_acc_overall_old_tasks_latest_model   = []

log_acc_overall_latest_new_tasks_original_model = []
log_acc_overall_latest_new_tasks_before_last_model = []
log_acc_overall_latest_new_tasks_latest_model   = []

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



for experiment_ID in range(experiments_number):

	# ======================================================================
	# == Retraining datasets generation
	# ======================================================================

	retraining_sets = New_instances_suite_1(n_retraining_sets*4, num_of_oversamples, \
									max_rotation,max_shift, \
									gn_dataset, blur_dataset, contrast_dataset)

	# retraining_sets = New_instances_suite_2(gn_dataset, blur_dataset, contrast_dataset)

	print("len(retraining_sets) = ", len(retraining_sets))


	# ======================================================================
	# == Declarations
	# ======================================================================
	Misclassification_threshold         = 50 # Set the threshold for retraining
	Misclassification_counter           = 0  # Counter to count number of mis-classifications
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

	digit_0_noise_1_counter = 0
	digit_1_noise_1_counter = 0
	digit_2_noise_1_counter = 0
	digit_3_noise_1_counter = 0
	digit_4_noise_1_counter = 0
	digit_5_noise_1_counter = 0
	digit_6_noise_1_counter = 0
	digit_7_noise_1_counter = 0
	digit_8_noise_1_counter = 0
	digit_9_noise_1_counter = 0

	digit_0_noise_2_counter = 0
	digit_1_noise_2_counter = 0
	digit_2_noise_2_counter = 0
	digit_3_noise_2_counter = 0
	digit_4_noise_2_counter = 0
	digit_5_noise_2_counter = 0
	digit_6_noise_2_counter = 0
	digit_7_noise_2_counter = 0
	digit_8_noise_2_counter = 0
	digit_9_noise_2_counter = 0


	i = 0 
	if Retraining_flag == True :

		for current_ID in range(n_retraining_sets):
			
			# Find a new instance from the set of generated retraining instances
			while True:
				# select retraining set
				retraining_set = retraining_sets[i]

				# assess its accuracy on last trained model
				acc = test(trained_models[-1], device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data

				# if accuracy below threshold then:
				if acc < Misclassification_threshold:

					# i)   set the retrainingset ID to the current ID
					retraining_set.ID = current_ID

					# ii)  append the retarining set the to the set of retrained sets
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
				# shuffle
				train_X_temp, train_Y_temp = shuffle(train_X_temp, train_Y_temp)

				noise_type_ID = 10   # mix of noise datasets
				dataset_ID = 1000000 # mix of noise datasets
				test_X_all_temp = None
				test_Y_all_temp = None
				accumilated_retraining_set = dset(dataset_ID, noise_type_ID,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all_temp, test_Y_all_temp)
		

				for lr_value_retrain in lr_learning_range:
					for ewc_lambda in ewc_lambda_range:

						ewc_lambdas = [ewc_lambda]

						#retrain
						print("==============================================================")
						print("Experiment ID = ",experiment_ID,", Retraining ID = ",current_ID,", oversamples_num = ",oversamples_num,", lr_value_retrain = ",lr_value_retrain,", ewc_lambda = ",ewc_lambda)
						print("==============================================================")

						retrained_model = Net().to(device)
						retrained_model.load_state_dict(trained_models[0].state_dict()) # Use initial model to do the retraining instead of the last retrainied model
						optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)

						for epoch in range(0, epoch_retrain):
							retrained_model, fisher_dict, optpar_dict = train_ewc(retrained_model, device, 1, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y, optimizer, epoch, ewc_lambdas, fisher_dict, optpar_dict)
						# fisher_dict, optpar_dict = on_task_update(0, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y, retrained_model, optimizer, fisher_dict, optpar_dict)
						# I commented out the above line as I am not doing EWC to remember the new instanes only the original trainining dataset.

						# Append to temp arrays to score later
						temp_models_array.append(retrained_model)
						temp_ovesamples_array.append(oversamples_num)
						temp_lr_retraining_array.append(lr_value_retrain)
						temp_ewc_array.append(ewc_lambda)

			end_time1 = time.time()

			# DONE TODO score the different models generated at this retraining ID and choose one to go forwards. 
			# loop over models
			temp_acc_original_data    = []
			temp_acc_accumilated_data = []

			for temp_model in temp_models_array:
				temp_acc_original_data.append(test(temp_model, device, X_test, Y_test))
				temp_acc_accumilated_data.append(test(temp_model, device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y))
			temp_acc_original_data = np.array(temp_acc_original_data)
			temp_acc_accumilated_data = np.array(temp_acc_accumilated_data)

			# # == Method 1 of scoring ==================================================
			# # Assess if they fit the criteria below:
			# original_data_acc_threshold = 95

			# # Original data performance must be above ---
			# temp_indicies = np.where(temp_acc_original_data >= original_data_acc_threshold)
			# temp_indicies = temp_indicies[0]

			# # If non then select the highest one performing on the original data
			# if len(temp_indicies) == 0:
			# 	temp_indicies = np.where(temp_acc_original_data == max(temp_acc_original_data)) 
			# 	temp_indicies = temp_indicies[0]
			
			# # Then out of the selected choose the one with the highest performance on the new data
			# accumilated_data_acc_filtered = temp_acc_accumilated_data[temp_indicies]
			# max_accumilated_data_acc = max(accumilated_data_acc_filtered)
			# max_index = np.where(accumilated_data_acc_filtered == max_accumilated_data_acc)
			# index_of_best_scoring_model = temp_indicies[max_index[0]][0]
			# # =========================================================================

			# == Method 2 of scoring ==================================================
			# Assess if they fit the criteria below:
			accumilated_data_acc_threshold = 70

			# accumilated data performance must be above ---
			temp_indicies = np.where(temp_acc_accumilated_data >= accumilated_data_acc_threshold)
			temp_indicies = temp_indicies[0]

			# If non then select the highest one performing on the accumilated data
			if len(temp_indicies) == 0:
				temp_indicies = np.where(temp_acc_accumilated_data == max(temp_acc_accumilated_data)) 
				temp_indicies = temp_indicies[0]
			
			# Then out of the selected choose the one with the highest performance on the new data
			original_data_acc_filtered = temp_acc_original_data[temp_indicies]
			max_original_data_acc = max(original_data_acc_filtered)
			max_index = np.where(original_data_acc_filtered == max_original_data_acc)
			index_of_best_scoring_model = temp_indicies[max_index[0]][0]
			# =========================================================================

			# # == Method 3 of scoring ==================================================

			
			# index_of_best_scoring_model = np.argmin(np.abs((temp_acc_original_data/temp_acc_accumilated_data) - 1))

										  
			
			
			# Based on the scoring choose which model to choose as the retrained model
			retrained_model = temp_models_array[index_of_best_scoring_model]

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

			elif retraining_set.noise_type == 1 and retraining_set.label == 0:
				digit_0_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 1:
				digit_1_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 2:
				digit_2_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 3:
				digit_3_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 4:
				digit_4_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 5:
				digit_5_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 6:
				digit_6_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 7:
				digit_7_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 8:
				digit_8_noise_1_counter += 1
			elif retraining_set.noise_type == 1 and retraining_set.label == 9:
				digit_9_noise_1_counter += 1

			elif retraining_set.noise_type == 2 and retraining_set.label == 0:
				digit_0_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 1:
				digit_1_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 2:
				digit_2_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 3:
				digit_3_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 4:
				digit_4_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 5:
				digit_5_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 6:
				digit_6_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 7:
				digit_7_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 8:
				digit_8_noise_2_counter += 1
			elif retraining_set.noise_type == 2 and retraining_set.label == 9:
				digit_9_noise_2_counter += 1



			# Creating accumilated tasks dataset based on the digit and noise types of the new instances involved in the accumilated retrainng dataset. 

			array_of_tasks_X       = np.array([gn_digit_0_test_X,gn_digit_1_test_X,gn_digit_2_test_X,gn_digit_3_test_X,gn_digit_4_test_X,gn_digit_5_test_X,gn_digit_6_test_X,gn_digit_7_test_X,gn_digit_8_test_X,gn_digit_9_test_X,\
									blur_digit_0_test_X,blur_digit_1_test_X,blur_digit_2_test_X,blur_digit_3_test_X,blur_digit_4_test_X,blur_digit_5_test_X,blur_digit_6_test_X,blur_digit_7_test_X,blur_digit_8_test_X,blur_digit_9_test_X,\
									contrast_digit_0_test_X,contrast_digit_1_test_X,contrast_digit_2_test_X,contrast_digit_3_test_X,contrast_digit_4_test_X,contrast_digit_5_test_X,contrast_digit_6_test_X,contrast_digit_7_test_X,contrast_digit_8_test_X,contrast_digit_9_test_X])

			array_of_tasks_Y       = np.array([gn_digit_0_test_Y,gn_digit_1_test_Y,gn_digit_2_test_Y,gn_digit_3_test_Y,gn_digit_4_test_Y,gn_digit_5_test_Y,gn_digit_6_test_Y,gn_digit_7_test_Y,gn_digit_8_test_Y,gn_digit_9_test_Y,\
									blur_digit_0_test_Y,blur_digit_1_test_Y,blur_digit_2_test_Y,blur_digit_3_test_Y,blur_digit_4_test_Y,blur_digit_5_test_Y,blur_digit_6_test_Y,blur_digit_7_test_Y,blur_digit_8_test_Y,blur_digit_9_test_Y,\
									contrast_digit_0_test_Y,contrast_digit_1_test_Y,contrast_digit_2_test_Y,contrast_digit_3_test_Y,contrast_digit_4_test_Y,contrast_digit_5_test_Y,contrast_digit_6_test_Y,contrast_digit_7_test_Y,contrast_digit_8_test_Y,contrast_digit_9_test_Y])

			array_of_tasks_flags = np.array([digit_0_noise_0_counter,digit_1_noise_0_counter,digit_2_noise_0_counter,digit_3_noise_0_counter,digit_4_noise_0_counter,digit_5_noise_0_counter,digit_6_noise_0_counter,digit_7_noise_0_counter,digit_8_noise_0_counter,digit_9_noise_0_counter,\
									digit_0_noise_1_counter,digit_1_noise_1_counter,digit_2_noise_1_counter,digit_3_noise_1_counter,digit_4_noise_1_counter,digit_5_noise_1_counter,digit_6_noise_1_counter,digit_7_noise_1_counter,digit_8_noise_1_counter,digit_9_noise_1_counter,\
									digit_0_noise_2_counter,digit_1_noise_2_counter,digit_2_noise_2_counter,digit_3_noise_2_counter,digit_4_noise_2_counter,digit_5_noise_2_counter,digit_6_noise_2_counter,digit_7_noise_2_counter,digit_8_noise_2_counter,digit_9_noise_2_counter])
			
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
			log_oversampling_size_array.append(temp_ovesamples_array[index_of_best_scoring_model])
			log_accumilated_retraining_dataset_size_array.append(len(accumilated_retraining_set.train_Y))
			log_lr_retraining_array.append(temp_lr_retraining_array[index_of_best_scoring_model])
			log_ewc_lambda_array.append(temp_ewc_array[index_of_best_scoring_model])

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
			
			# TODO populate the second set logging arrays
			# populat the digita array too, do not forget
			log_added_instance_digit_array.append(retraining_set.label)
			log_acc_digit_0_original.append(test(trained_models[-1], device, digit_0_test_X, digit_0_test_Y))
			log_acc_digit_1_original.append(test(trained_models[-1], device, digit_1_test_X, digit_1_test_Y))
			log_acc_digit_2_original.append(test(trained_models[-1], device, digit_2_test_X, digit_2_test_Y))
			log_acc_digit_3_original.append(test(trained_models[-1], device, digit_3_test_X, digit_3_test_Y))
			log_acc_digit_4_original.append(test(trained_models[-1], device, digit_4_test_X, digit_4_test_Y))
			log_acc_digit_5_original.append(test(trained_models[-1], device, digit_5_test_X, digit_5_test_Y))
			log_acc_digit_6_original.append(test(trained_models[-1], device, digit_6_test_X, digit_6_test_Y))
			log_acc_digit_7_original.append(test(trained_models[-1], device, digit_7_test_X, digit_7_test_Y))
			log_acc_digit_8_original.append(test(trained_models[-1], device, digit_8_test_X, digit_8_test_Y))
			log_acc_digit_9_original.append(test(trained_models[-1], device, digit_9_test_X, digit_9_test_Y))

			log_acc_digit_0_noise_0.append(test(trained_models[-1], device, gn_digit_0_test_X, gn_digit_0_test_Y))
			log_acc_digit_0_noise_0_flag.append(digit_0_noise_0_counter)
			log_acc_digit_1_noise_0.append(test(trained_models[-1], device, gn_digit_1_test_X, gn_digit_1_test_Y))
			log_acc_digit_1_noise_0_flag.append(digit_1_noise_0_counter)
			log_acc_digit_2_noise_0.append(test(trained_models[-1], device, gn_digit_2_test_X, gn_digit_2_test_Y))
			log_acc_digit_2_noise_0_flag.append(digit_2_noise_0_counter)
			log_acc_digit_3_noise_0.append(test(trained_models[-1], device, gn_digit_3_test_X, gn_digit_3_test_Y))
			log_acc_digit_3_noise_0_flag.append(digit_3_noise_0_counter)
			log_acc_digit_4_noise_0.append(test(trained_models[-1], device, gn_digit_4_test_X, gn_digit_4_test_Y))
			log_acc_digit_4_noise_0_flag.append(digit_4_noise_0_counter)
			log_acc_digit_5_noise_0.append(test(trained_models[-1], device, gn_digit_5_test_X, gn_digit_5_test_Y))
			log_acc_digit_5_noise_0_flag.append(digit_5_noise_0_counter)
			log_acc_digit_6_noise_0.append(test(trained_models[-1], device, gn_digit_6_test_X, gn_digit_6_test_Y))
			log_acc_digit_6_noise_0_flag.append(digit_6_noise_0_counter)
			log_acc_digit_7_noise_0.append(test(trained_models[-1], device, gn_digit_7_test_X, gn_digit_7_test_Y))
			log_acc_digit_7_noise_0_flag.append(digit_7_noise_0_counter)
			log_acc_digit_8_noise_0.append(test(trained_models[-1], device, gn_digit_8_test_X, gn_digit_8_test_Y))
			log_acc_digit_8_noise_0_flag.append(digit_8_noise_0_counter)
			log_acc_digit_9_noise_0.append(test(trained_models[-1], device, gn_digit_9_test_X, gn_digit_9_test_Y))
			log_acc_digit_9_noise_0_flag.append(digit_9_noise_0_counter)

			log_acc_digit_0_noise_1.append(test(trained_models[-1], device, blur_digit_0_test_X, blur_digit_0_test_Y))
			log_acc_digit_0_noise_1_flag.append(digit_0_noise_1_counter)
			log_acc_digit_1_noise_1.append(test(trained_models[-1], device, blur_digit_1_test_X, blur_digit_1_test_Y))
			log_acc_digit_1_noise_1_flag.append(digit_1_noise_1_counter)
			log_acc_digit_2_noise_1.append(test(trained_models[-1], device, blur_digit_2_test_X, blur_digit_2_test_Y))
			log_acc_digit_2_noise_1_flag.append(digit_2_noise_1_counter)
			log_acc_digit_3_noise_1.append(test(trained_models[-1], device, blur_digit_3_test_X, blur_digit_3_test_Y))
			log_acc_digit_3_noise_1_flag.append(digit_3_noise_1_counter)
			log_acc_digit_4_noise_1.append(test(trained_models[-1], device, blur_digit_4_test_X, blur_digit_4_test_Y))
			log_acc_digit_4_noise_1_flag.append(digit_4_noise_1_counter)
			log_acc_digit_5_noise_1.append(test(trained_models[-1], device, blur_digit_5_test_X, blur_digit_5_test_Y))
			log_acc_digit_5_noise_1_flag.append(digit_5_noise_1_counter)
			log_acc_digit_6_noise_1.append(test(trained_models[-1], device, blur_digit_6_test_X, blur_digit_6_test_Y))
			log_acc_digit_6_noise_1_flag.append(digit_6_noise_1_counter)
			log_acc_digit_7_noise_1.append(test(trained_models[-1], device, blur_digit_7_test_X, blur_digit_7_test_Y))
			log_acc_digit_7_noise_1_flag.append(digit_7_noise_1_counter)
			log_acc_digit_8_noise_1.append(test(trained_models[-1], device, blur_digit_8_test_X, blur_digit_8_test_Y))
			log_acc_digit_8_noise_1_flag.append(digit_8_noise_1_counter)
			log_acc_digit_9_noise_1.append(test(trained_models[-1], device, blur_digit_9_test_X, blur_digit_9_test_Y))
			log_acc_digit_9_noise_1_flag.append(digit_9_noise_1_counter)

			log_acc_digit_0_noise_2.append(test(trained_models[-1], device, contrast_digit_0_test_X, contrast_digit_0_test_Y))
			log_acc_digit_0_noise_2_flag.append(digit_0_noise_2_counter)
			log_acc_digit_1_noise_2.append(test(trained_models[-1], device, contrast_digit_1_test_X, contrast_digit_1_test_Y))
			log_acc_digit_1_noise_2_flag.append(digit_1_noise_2_counter)
			log_acc_digit_2_noise_2.append(test(trained_models[-1], device, contrast_digit_2_test_X, contrast_digit_2_test_Y))
			log_acc_digit_2_noise_2_flag.append(digit_2_noise_2_counter)
			log_acc_digit_3_noise_2.append(test(trained_models[-1], device, contrast_digit_3_test_X, contrast_digit_3_test_Y))
			log_acc_digit_3_noise_2_flag.append(digit_3_noise_2_counter)
			log_acc_digit_4_noise_2.append(test(trained_models[-1], device, contrast_digit_4_test_X, contrast_digit_4_test_Y))
			log_acc_digit_4_noise_2_flag.append(digit_4_noise_2_counter)
			log_acc_digit_5_noise_2.append(test(trained_models[-1], device, contrast_digit_5_test_X, contrast_digit_5_test_Y))
			log_acc_digit_5_noise_2_flag.append(digit_5_noise_2_counter)
			log_acc_digit_6_noise_2.append(test(trained_models[-1], device, contrast_digit_6_test_X, contrast_digit_6_test_Y))
			log_acc_digit_6_noise_2_flag.append(digit_6_noise_2_counter)
			log_acc_digit_7_noise_2.append(test(trained_models[-1], device, contrast_digit_7_test_X, contrast_digit_7_test_Y))
			log_acc_digit_7_noise_2_flag.append(digit_7_noise_2_counter)
			log_acc_digit_8_noise_2.append(test(trained_models[-1], device, contrast_digit_8_test_X, contrast_digit_8_test_Y))
			log_acc_digit_8_noise_2_flag.append(digit_8_noise_2_counter)
			log_acc_digit_9_noise_2.append(test(trained_models[-1], device, contrast_digit_9_test_X, contrast_digit_9_test_Y))
			log_acc_digit_9_noise_2_flag.append(digit_9_noise_2_counter)

			log_acc_overall_old_tasks_original_model.append(test(trained_models[0], device, X_test, Y_test))
			log_acc_overall_old_tasks_before_last_model.append(test(trained_models[-2], device, X_test, Y_test))
			log_acc_overall_old_tasks_latest_model.append(test(trained_models[-1], device, X_test, Y_test))

			log_acc_overall_latest_new_tasks_original_model.append(test(trained_models[0], device, accumilated_new_tasks_involved_from_new_instances_test_dataset_X, accumilated_new_tasks_involved_from_new_instances_test_dataset_Y))
			log_acc_overall_latest_new_tasks_before_last_model.append(test(trained_models[-2], device, accumilated_new_tasks_involved_from_new_instances_test_dataset_X, accumilated_new_tasks_involved_from_new_instances_test_dataset_Y))
			log_acc_overall_latest_new_tasks_latest_model.append(test(trained_models[-1], device, accumilated_new_tasks_involved_from_new_instances_test_dataset_X, accumilated_new_tasks_involved_from_new_instances_test_dataset_Y))

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
					temp_task_test_X = gn_digit_0_test_X
					temp_task_test_Y = gn_digit_0_test_Y
				elif re_set.noise_type == 0 and re_set.label == 1:
					temp_task_test_X = gn_digit_1_test_X
					temp_task_test_Y = gn_digit_1_test_Y
				elif re_set.noise_type == 0 and re_set.label == 2:
					temp_task_test_X = gn_digit_2_test_X
					temp_task_test_Y = gn_digit_2_test_Y
				elif re_set.noise_type == 0 and re_set.label == 3:
					temp_task_test_X = gn_digit_3_test_X
					temp_task_test_Y = gn_digit_3_test_Y
				elif re_set.noise_type == 0 and re_set.label == 4:
					temp_task_test_X = gn_digit_4_test_X
					temp_task_test_Y = gn_digit_4_test_Y
				elif re_set.noise_type == 0 and re_set.label == 5:
					temp_task_test_X = gn_digit_5_test_X
					temp_task_test_Y = gn_digit_5_test_Y
				elif re_set.noise_type == 0 and re_set.label == 6:
					temp_task_test_X = gn_digit_6_test_X
					temp_task_test_Y = gn_digit_6_test_Y
				elif re_set.noise_type == 0 and re_set.label == 7:
					temp_task_test_X = gn_digit_7_test_X
					temp_task_test_Y = gn_digit_7_test_Y
				elif re_set.noise_type == 0 and re_set.label == 8:
					temp_task_test_X = gn_digit_8_test_X
					temp_task_test_Y = gn_digit_8_test_Y
				elif re_set.noise_type == 0 and re_set.label == 9:
					temp_task_test_X = gn_digit_9_test_X
					temp_task_test_Y = gn_digit_9_test_Y

				elif re_set.noise_type == 1 and re_set.label == 0:
					temp_task_test_X = blur_digit_0_test_X
					temp_task_test_Y = blur_digit_0_test_Y
				elif re_set.noise_type == 1 and re_set.label == 1:
					temp_task_test_X = blur_digit_1_test_X
					temp_task_test_Y = blur_digit_1_test_Y
				elif re_set.noise_type == 1 and re_set.label == 2:
					temp_task_test_X = blur_digit_2_test_X
					temp_task_test_Y = blur_digit_2_test_Y
				elif re_set.noise_type == 1 and re_set.label == 3:
					temp_task_test_X = blur_digit_3_test_X
					temp_task_test_Y = blur_digit_3_test_Y
				elif re_set.noise_type == 1 and re_set.label == 4:
					temp_task_test_X = blur_digit_4_test_X
					temp_task_test_Y = blur_digit_4_test_Y
				elif re_set.noise_type == 1 and re_set.label == 5:
					temp_task_test_X = blur_digit_5_test_X
					temp_task_test_Y = blur_digit_5_test_Y
				elif re_set.noise_type == 1 and re_set.label == 6:
					temp_task_test_X = blur_digit_6_test_X
					temp_task_test_Y = blur_digit_6_test_Y
				elif re_set.noise_type == 1 and re_set.label == 7:
					temp_task_test_X = blur_digit_7_test_X
					temp_task_test_Y = blur_digit_7_test_Y
				elif re_set.noise_type == 1 and re_set.label == 8:
					temp_task_test_X = blur_digit_8_test_X
					temp_task_test_Y = blur_digit_8_test_Y
				elif re_set.noise_type == 1 and re_set.label == 9:
					temp_task_test_X = blur_digit_9_test_X
					temp_task_test_Y = blur_digit_9_test_Y

				elif re_set.noise_type == 2 and re_set.label == 0:
					temp_task_test_X = contrast_digit_0_test_X
					temp_task_test_Y = contrast_digit_0_test_Y
				elif re_set.noise_type == 2 and re_set.label == 1:
					temp_task_test_X = contrast_digit_1_test_X
					temp_task_test_Y = contrast_digit_1_test_Y
				elif re_set.noise_type == 2 and re_set.label == 2:
					temp_task_test_X = contrast_digit_2_test_X
					temp_task_test_Y = contrast_digit_2_test_Y
				elif re_set.noise_type == 2 and re_set.label == 3:
					temp_task_test_X = contrast_digit_3_test_X
					temp_task_test_Y = contrast_digit_3_test_Y
				elif re_set.noise_type == 2 and re_set.label == 4:
					temp_task_test_X = contrast_digit_4_test_X
					temp_task_test_Y = contrast_digit_4_test_Y
				elif re_set.noise_type == 2 and re_set.label == 5:
					temp_task_test_X = contrast_digit_5_test_X
					temp_task_test_Y = contrast_digit_5_test_Y
				elif re_set.noise_type == 2 and re_set.label == 6:
					temp_task_test_X = contrast_digit_6_test_X
					temp_task_test_Y = contrast_digit_6_test_Y
				elif re_set.noise_type == 2 and re_set.label == 7:
					temp_task_test_X = contrast_digit_7_test_X
					temp_task_test_Y = contrast_digit_7_test_Y
				elif re_set.noise_type == 2 and re_set.label == 8:
					temp_task_test_X = contrast_digit_8_test_X
					temp_task_test_Y = contrast_digit_8_test_Y
				elif re_set.noise_type == 2 and re_set.label == 9:
					temp_task_test_X = contrast_digit_9_test_X
					temp_task_test_Y = contrast_digit_9_test_Y
				

				log_acc_on_task_first_model.append(test(trained_models[0], device, temp_task_test_X, temp_task_test_Y))
				log_acc_on_task_before_last_model.append(test(trained_models[-2], device, temp_task_test_X, temp_task_test_Y))
				log_acc_on_task_last_model.append(test(trained_models[-1], device, temp_task_test_X, temp_task_test_Y))
				

# Write log file

# Folder "results" if not already there
output_folder = "Results_logs"
file_name1     = "results_combined_15-1.csv"
file_name2     = "results_combined_15-2.csv"
file_name3     = "results_combined_15-3.csv"

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
					last_model_acc_on_accumilated_oversampled_new_instances_training_data, last_model_acc_on_accumilated_oversampled_new_instances_testing_data\n')

	for i in range(len(log_current_ID_array)):
		log_file.write('%d,%d,%d,%d,%d,%d,%1.6f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n' %\
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
					log_last_model_acc_on_accumilated_oversampled_new_instances_training_data[i],log_last_model_acc_on_accumilated_oversampled_new_instances_testing_data[i]))

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
					acc_digit_0_original, \
					acc_digit_1_original, \
					acc_digit_2_original, \
					acc_digit_3_original, \
					acc_digit_4_original, \
					acc_digit_5_original, \
					acc_digit_6_original, \
					acc_digit_7_original, \
					acc_digit_8_original, \
					acc_digit_9_original, \
					acc_digit_0_noise_0, \
					\
					digit_0_noise_0_counter,\
					acc_digit_1_noise_0, \
					digit_1_noise_0_counter,\
					acc_digit_2_noise_0, \
					digit_2_noise_0_counter,\
					acc_digit_3_noise_0, \
					digit_3_noise_0_counter,\
					acc_digit_4_noise_0, \
					digit_4_noise_0_counter,\
					acc_digit_5_noise_0, \
					digit_5_noise_0_counter,\
					acc_digit_6_noise_0, \
					digit_6_noise_0_counter,\
					acc_digit_7_noise_0, \
					digit_7_noise_0_counter,\
					acc_digit_8_noise_0, \
					digit_8_noise_0_counter,\
					acc_digit_9_noise_0, \
					digit_9_noise_0_counter,\
					\
					acc_digit_0_noise_1, \
					digit_0_noise_1_counter,\
					acc_digit_1_noise_1, \
					digit_1_noise_1_counter,\
					acc_digit_2_noise_1, \
					digit_2_noise_1_counter,\
					acc_digit_3_noise_1, \
					digit_3_noise_1_counter,\
					acc_digit_4_noise_1, \
					digit_4_noise_1_counter,\
					acc_digit_5_noise_1, \
					digit_5_noise_1_counter,\
					acc_digit_6_noise_1, \
					digit_6_noise_1_counter,\
					acc_digit_7_noise_1, \
					digit_7_noise_1_counter,\
					acc_digit_8_noise_1, \
					digit_8_noise_1_counter,\
					acc_digit_9_noise_1, \
					digit_9_noise_1_counter,\
					\
					acc_digit_0_noise_2, \
					digit_0_noise_2_counter,\
					acc_digit_1_noise_2, \
					digit_1_noise_2_counter,\
					acc_digit_2_noise_2, \
					digit_2_noise_2_counter,\
					acc_digit_3_noise_2, \
					digit_3_noise_2_counter,\
					acc_digit_4_noise_2, \
					digit_4_noise_2_counter,\
					acc_digit_5_noise_2, \
					digit_5_noise_2_counter,\
					acc_digit_6_noise_2, \
					digit_6_noise_2_counter,\
					acc_digit_7_noise_2, \
					digit_7_noise_2_counter,\
					acc_digit_8_noise_2, \
					digit_8_noise_2_counter,\
					acc_digit_9_noise_2, \
					digit_9_noise_2_counter,\
					\
					acc_overall_old_tasks_original_model,\
					acc_overall_old_tasks_before_last_model,\
					acc_overall_old_tasks_latest_model,\
					acc_overall_latest_new_tasks_original_model,\
					acc_overall_latest_new_tasks_before_last_model,\
					acc_overall_latest_new_tasks_latest_model\
					\n')

	for i in range(len(log_current_ID_array)):
		log_file.write('%d,%d,%d,%d,\
						%d,%d,%d,%1.6f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,\
						%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n' %\
					(log_experiment_ID_array[i], log_current_ID_array[i], log_added_instance_digit_array[i], log_added_instance_noise_type_array[i],\
					log_initial_dataset_size_array[i],log_oversampling_size_array[i], log_accumilated_retraining_dataset_size_array[i], log_lr_retraining_array[i],log_ewc_lambda_array[i],\
					log_acc_digit_0_original[i],\
					log_acc_digit_1_original[i],\
					log_acc_digit_2_original[i],\
					log_acc_digit_3_original[i],\
					log_acc_digit_4_original[i],\
					log_acc_digit_5_original[i],\
					log_acc_digit_6_original[i],\
					log_acc_digit_7_original[i],\
					log_acc_digit_8_original[i],\
					log_acc_digit_9_original[i],\

					log_acc_digit_0_noise_0[i],\
					log_acc_digit_0_noise_0_flag[i],\
					log_acc_digit_1_noise_0[i],\
					log_acc_digit_1_noise_0_flag[i],\
					log_acc_digit_2_noise_0[i],\
					log_acc_digit_2_noise_0_flag[i],\
					log_acc_digit_3_noise_0[i],\
					log_acc_digit_3_noise_0_flag[i],\
					log_acc_digit_4_noise_0[i],\
					log_acc_digit_4_noise_0_flag[i],\
					log_acc_digit_5_noise_0[i],\
					log_acc_digit_5_noise_0_flag[i],\
					log_acc_digit_6_noise_0[i],\
					log_acc_digit_6_noise_0_flag[i],\
					log_acc_digit_7_noise_0[i],\
					log_acc_digit_7_noise_0_flag[i],\
					log_acc_digit_8_noise_0[i],\
					log_acc_digit_8_noise_0_flag[i],\
					log_acc_digit_9_noise_0[i],\
					log_acc_digit_9_noise_0_flag[i],\
					
					log_acc_digit_0_noise_1[i],\
					log_acc_digit_0_noise_1_flag[i],\
					log_acc_digit_1_noise_1[i],\
					log_acc_digit_1_noise_1_flag[i],\
					log_acc_digit_2_noise_1[i],\
					log_acc_digit_2_noise_1_flag[i],\
					log_acc_digit_3_noise_1[i],\
					log_acc_digit_3_noise_1_flag[i],\
					log_acc_digit_4_noise_1[i],\
					log_acc_digit_4_noise_1_flag[i],\
					log_acc_digit_5_noise_1[i],\
					log_acc_digit_5_noise_1_flag[i],\
					log_acc_digit_6_noise_1[i],\
					log_acc_digit_6_noise_1_flag[i],\
					log_acc_digit_7_noise_1[i],\
					log_acc_digit_7_noise_1_flag[i],\
					log_acc_digit_8_noise_1[i],\
					log_acc_digit_8_noise_1_flag[i],\
					log_acc_digit_9_noise_1[i],\
					log_acc_digit_9_noise_1_flag[i],\

					log_acc_digit_0_noise_2[i],\
					log_acc_digit_0_noise_2_flag[i],\
					log_acc_digit_1_noise_2[i],\
					log_acc_digit_1_noise_2_flag[i],\
					log_acc_digit_2_noise_2[i],\
					log_acc_digit_2_noise_2_flag[i],\
					log_acc_digit_3_noise_2[i],\
					log_acc_digit_3_noise_2_flag[i],\
					log_acc_digit_4_noise_2[i],\
					log_acc_digit_4_noise_2_flag[i],\
					log_acc_digit_5_noise_2[i],\
					log_acc_digit_5_noise_2_flag[i],\
					log_acc_digit_6_noise_2[i],\
					log_acc_digit_6_noise_2_flag[i],\
					log_acc_digit_7_noise_2[i],\
					log_acc_digit_7_noise_2_flag[i],\
					log_acc_digit_8_noise_2[i],\
					log_acc_digit_8_noise_2_flag[i],\
					log_acc_digit_9_noise_2[i],\
					log_acc_digit_9_noise_2_flag[i],\

					log_acc_overall_old_tasks_original_model[i],\
					log_acc_overall_old_tasks_before_last_model[i],\
					log_acc_overall_old_tasks_latest_model[i],\
					log_acc_overall_latest_new_tasks_original_model[i],\
					log_acc_overall_latest_new_tasks_before_last_model[i],\
					log_acc_overall_latest_new_tasks_latest_model[i]))



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
log_instance_digit