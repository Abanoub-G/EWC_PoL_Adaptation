import random
# random.seed(0)

import numpy as np
from scripts.oversample import shift_oversampling, rotation_oversampling, shift_rotation_oversampling

class dset():
    def __init__(self,ID, noise_type,train_X, train_Y, test_X, test_Y,test_X_all,test_Y_all):
        self.ID        = ID
        self.noise_type = noise_type
        self.label      = train_Y[0]
        self.train_X    = train_X
        self.train_Y    = train_Y
        self.test_X     = test_X  # test data for same noise function but for other elements of the same class retrained on (to see if it generalises other rare elements of the same class)
        self.test_Y     = test_Y
        self.test_X_all = test_X_all  # test data for same noise function but for all elements different classes (to see if it generalises to all classes)
        self.test_Y_all = test_Y_all

    # def forward(self, x):
       

def New_instances_suite_1(n,n_oversamples,max_rotation,max_shift,*arg):
    # n              : Number of differnt retraining sets wanted
    # n_oversamples  : Number of oversamples wanted in each retraining set
    # max_rotation   : max rotation wanted in oversamples
    # max_shift      : max shift wanted in oversamples
    # *arg           : enter different datasets i.e. following different patterns
    
    # no_of_noise_patterns : Number of noise patterns supplied
    # n_per_noise          : Number of different retraining sets to be generated from each noise function
    
    no_of_noise_patterns = len(arg)

    if n%no_of_noise_patterns == 0:
        n_per_noise = n / no_of_noise_patterns
    else:
        n_per_noise = (n - n%no_of_noise_patterns) / no_of_noise_patterns
    # print("n_per_noise = ", n_per_noise)
    n_per_noise = int(n_per_noise)
    # print("n_per_noise = ", n_per_noise)
    
    remainder_cover_counter = 0
    
    # no_of_classes        = len(datasets[0])
    retraining_sets      = []
    ID_counter           = 0 


    for i in range(no_of_noise_patterns):

        dataset = arg[i]

        train_X    = dataset["training_images"] # training data
        train_Y    = dataset["training_labels"] # training labels
        test_X  = test_X_all  = dataset["testing_images"] # testing data
        test_Y  = test_Y_all  = dataset["testing_labels"] # testing labels
        

        # This code below tries to cover up for the remainder from the noise patterns
        if n%no_of_noise_patterns != 0  and  n%no_of_noise_patterns > remainder_cover_counter:
            remainder_cover = 1
            remainder_cover_counter += 1
        else:
            remainder_cover = 0


        for j in random.sample(range(len(train_Y)), n_per_noise+remainder_cover):
            # randomly select from list and make sure not to repeat
            instance = train_X[j]
            instance_label = train_Y[j]

            # oversample instance
            train_X_temp, train_Y_temp = shift_rotation_oversampling(instance, instance_label, \
                                                            n_oversamples, max_shift, max_rotation)
            
            # oversample instance to generate a set of instances (1/4 * the number of instances used) to use as a test dataset
            test_X_temp, test_Y_temp   = shift_rotation_oversampling(instance, instance_label, \
                                                            int(n_oversamples/4), max_shift, max_rotation)


            ID         = ID_counter

            noise_type = i

            ID_counter += 1 

            print("ID_counter = ", ID_counter)
            print("noise_type = ", noise_type)
            print("train_X_temp.dtype = ", train_X_temp.dtype)
            print("train_X_temp.shape = ", train_X_temp.shape)
            print("train_Y_temp.dtype = ", train_Y_temp.dtype)
            print("train_Y_temp.shape = ", train_Y_temp.shape)
            print("test_X_temp.dtype = ", test_X_temp.dtype)
            print("test_X_temp.shape = ", test_X_temp.shape)
            print("test_Y_temp.dtype = ", test_Y_temp.dtype)
            print("test_Y_temp.shape = ", test_Y_temp.shape)
            # input("please press")

            retraining_sets.append(dset(ID, noise_type,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all, test_Y_all))

    # shuffle retraining sets 
    random.shuffle(retraining_sets)

    # renumber the IDs for each set in the retraining_sets array so that it becomes in order from 1 to --.
    for i in range(len(retraining_sets)):
        retraining_sets[i].ID = i

    print("Done generating New isntances")
    return retraining_sets



def New_instances_suite_2(*arg):
    no_of_noise_patterns = len(arg)
    # no_of_classes        = len(datasets[0])
    retraining_sets      = []
    ID_counter           = 0 
    for i in range(no_of_noise_patterns):
        dataset = arg[i]
        # no_of_classes = len(dataset)
        train_X    = dataset["training_images"] # training data
        train_Y    = dataset["training_labels"] # training labels
        test_X  = test_X_all  = dataset["testing_images"] # testing data
        test_Y  = test_Y_all  = dataset["testing_labels"] # testing labels
        # print("train_Y", train_Y )
        
        for digit in np.unique(train_Y):

            # get indiceis of all training point for this digit
            indicies = np.where(train_Y == digit)
            indicies = indicies[0]
            # print("digit = ", digit)
            train_X_temp = train_X[indicies]
            train_Y_temp = train_Y[indicies]

            indicies = np.where(test_Y == digit)
            indicies = indicies[0]

            test_X_temp = test_X[indicies]
            test_Y_temp = test_Y[indicies]

            ID         = ID_counter

            noise_type = i

            ID_counter += 1 

            # print("train_X_temp.dtype = ", train_X_temp.dtype)
            # print("train_X_temp.shape = ", train_X_temp.shape)
            # print("train_Y_temp.dtype = ", train_Y_temp.dtype)
            # print("train_Y_temp.shape = ", train_Y_temp.shape)
            # print("test_X_temp.dtype = ", test_X_temp.dtype)
            # print("test_X_temp.shape = ", test_X_temp.shape)
            # print("test_Y_temp.dtype = ", test_Y_temp.dtype)
            # print("test_Y_temp.shape = ", test_Y_temp.shape)
            # input("please press")


            retraining_sets.append(dset(ID, noise_type,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all, test_Y_all))

    # shuffle retraining sets 
    random.shuffle(retraining_sets)

    # renumber the IDs for each set in the retraining_sets array so that it becomes in order from 1 to --.
    for i in range(len(retraining_sets)):
        retraining_sets[i].ID = i

    return retraining_sets


def New_instances_suite_3(num_of_oversamples_per_task, *arg):
    no_of_noise_patterns = len(arg)
    # no_of_classes        = len(datasets[0])
    retraining_sets      = []
    ID_counter           = 0 
    for i in range(no_of_noise_patterns):
        dataset = arg[i]
        # no_of_classes = len(dataset)
        train_X    = dataset["training_images"] # training data
        train_Y    = dataset["training_labels"] # training labels
        test_X  = test_X_all  = dataset["testing_images"] # testing data
        test_Y  = test_Y_all  = dataset["testing_labels"] # testing labels
        # print("train_Y", train_Y )
        
        for digit in np.unique(train_Y):

            # get indiceis of all training point for this digit
            # indicies = np.where(train_Y == digit)
            # indicies = indicies[0]
            # print("digit = ", digit)
            # train_X_temp = train_X[indicies]
            # train_Y_temp = train_Y[indicies]
            train_X_temp = train_X[np.where(train_Y == digit)]
            train_Y_temp = train_Y[np.where(train_Y == digit)]

            # indicies = np.where(test_Y == digit)
            # indicies = indicies[0]

            # test_X_temp = test_X[indicies]
            # test_Y_temp = test_Y[indicies]
            test_X_temp = test_X[np.where(test_Y == digit)]
            test_Y_temp = test_Y[np.where(test_Y == digit)]

            # Setting the required number of oversamples
            train_X_temp = train_X_temp[:num_of_oversamples_per_task]
            train_Y_temp = train_Y_temp[:num_of_oversamples_per_task]


            ID         = ID_counter

            noise_type = i

            ID_counter += 1 

            # print("train_X_temp.dtype = ", train_X_temp.dtype)
            # print("train_X_temp.shape = ", train_X_temp.shape)
            # print("train_Y_temp.dtype = ", train_Y_temp.dtype)
            # print("train_Y_temp.shape = ", train_Y_temp.shape)
            # print("test_X_temp.dtype = ", test_X_temp.dtype)
            # print("test_X_temp.shape = ", test_X_temp.shape)
            # print("test_Y_temp.dtype = ", test_Y_temp.dtype)
            # print("test_Y_temp.shape = ", test_Y_temp.shape)
            # input("please press")


            retraining_sets.append(dset(ID, noise_type,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all, test_Y_all))

    # shuffle retraining sets 
    random.shuffle(retraining_sets)

    # renumber the IDs for each set in the retraining_sets array so that it becomes in order from 1 to --.
    for i in range(len(retraining_sets)):
        retraining_sets[i].ID = i

    return retraining_sets


def New_instances_suite_4(n,n_oversamples,max_rotation,max_shift,*arg):
    # n              : Number of differnt retraining sets wanted
    # n_oversamples  : Number of oversamples wanted in each retraining set
    # max_rotation   : max rotation wanted in oversamples
    # max_shift      : max shift wanted in oversamples
    # *arg           : enter different tasks to generate new instances from 
    
    # no_of_tasks : Number of tasks supplied
    # n_per_task          : Number of different retraining sets to be generated from each task
    
    no_of_tasks = len(arg)
    print(no_of_tasks)
    # no_of_noise_patterns = len(arg)

    if n%no_of_tasks == 0:
        n_per_task = n / no_of_tasks
    else:
        n_per_task = (n - n%no_of_tasks) / no_of_tasks
    # print("n_per_task = ", n_per_task)
    n_per_task = int(n_per_task)
    # print("n_per_noise = ", n_per_noise)
    
    remainder_cover_counter = 0
    
    # no_of_classes        = len(datasets[0])
    retraining_sets      = []
    ID_counter           = 0 


    for i in range(no_of_tasks):

        task = arg[i]
        # print(task)
        # print(len(task))
        # input("press enter to continue")

        train_X               = task[0]#dataset["training_images"] # training data
        train_Y               = task[1]#dataset["training_labels"] # training labels
        test_X  = test_X_all  = task[2]#dataset["testing_images"] # testing data
        test_Y  = test_Y_all  = task[3]#dataset["testing_labels"] # testing labels
        

        # This code below tries to cover up for the remainder from the noise patterns
        if n%no_of_tasks != 0  and  n%no_of_tasks > remainder_cover_counter:
            remainder_cover = 1
            remainder_cover_counter += 1
        else:
            remainder_cover = 0


        for j in random.sample(range(len(train_Y)), n_per_task+remainder_cover):
            # randomly select from list and make sure not to repeat
            instance = train_X[j]
            instance_label = train_Y[j]

            # oversample instance
            train_X_temp, train_Y_temp = shift_rotation_oversampling(instance, instance_label, \
                                                            n_oversamples, max_shift, max_rotation)
            
            # oversample instance to generate a set of instances (1/4 * the number of instances used) to use as a test dataset
            test_X_temp, test_Y_temp   = shift_rotation_oversampling(instance, instance_label, \
                                                            int(n_oversamples/4), max_shift, max_rotation)

            # print("train_X_temp.dtype = ", train_X_temp.dtype)
            # print("train_X_temp.shape = ", train_X_temp.shape)
            # print("train_Y_temp.dtype = ", train_Y_temp.dtype)
            # print("train_Y_temp.shape = ", train_Y_temp.shape)
            # print("test_X_temp.dtype = ", test_X_temp.dtype)
            # print("test_X_temp.shape = ", test_X_temp.shape)
            # print("test_Y_temp.dtype = ", test_Y_temp.dtype)
            # print("test_Y_temp.shape = ", test_Y_temp.shape)
            # input("please press")

            ID         = ID_counter

            if i == 0:
                noise_type = 0
            if i > 0:
                noise_type =1
            if i > 6:
                noise_type = 2

            # noise_type = i

            ID_counter += 1 

            retraining_sets.append(dset(ID, noise_type,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all, test_Y_all))
            print("ID_counter = ", ID_counter)
            print("noise_type = ", noise_type)

            print("train_X_temp.dtype = ", train_X_temp.dtype)
            print("train_X_temp.shape = ", train_X_temp.shape)
            print("train_Y_temp.dtype = ", train_Y_temp.dtype)
            print("train_Y_temp.shape = ", train_Y_temp.shape)
            print("test_X_temp.dtype = ", test_X_temp.dtype)
            print("test_X_temp.shape = ", test_X_temp.shape)
            print("test_Y_temp.dtype = ", test_Y_temp.dtype)
            print("test_Y_temp.shape = ", test_Y_temp.shape)
            # input("please press")
            # print("type(train_X_temp) = ",type(train_X_temp))
            # print("train_X_temp.shape = ", train_X_temp.shape)

            # print("type(train_Y_temp) = ",type(train_Y_temp))
            # print("train_Y_temp.shape = ", train_Y_temp.shape)

            # print("type(test_X_temp) = ",type(test_X_temp))
            # print("test_X_temp.shape = ", test_X_temp.shape)

            # print("type(test_Y_temp) = ",type(test_Y_temp))
            # print("test_Y_temp.shape = ", test_Y_temp.shape)
       
            # input("========= press enter to continue===============" )

            # print each of the train X temp and train y temp to see wher the multi target issue is coming from.

    # shuffle retraining sets 
    random.shuffle(retraining_sets)

    # renumber the IDs for each set in the retraining_sets array so that it becomes in order from 1 to --.
    for i in range(len(retraining_sets)):
        retraining_sets[i].ID = i

    print("Done generating New isntances")
    return retraining_sets

