'''
Created on Sep,2022
'''


import numpy as np
import time
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from algo_utils import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot
from transf_functions import get_trans_function



class AdBetWOA():
    def __init__(self, args, features_train, features_test):

        self.num_agents = args.pop_size
        self.max_iter = args.epoch_algo
        self.trans_function_shape = 's'
        self.obj_function = compute_fitness
        self.classifier = 'KNN'

        self.train_data = features_train[:,:-1]
        self.train_label = features_train[:,-1]

        self.test_data = features_test[:,:-1]
        self.test_label = features_test[:,-1]

        self.save_conv_graph = True

   
    def WOA(self):

        agent_name = 'whale'
        num_features = self.train_data.shape[1]
        trans_function = get_trans_function(self.trans_function_shape)

        # setting up the objectives
        weight_acc = None
        if(self.obj_function==compute_fitness):
            weight_acc = float(input('Weight for the classification accuracy [0-1]: '))
        
        obj = (self.obj_function, weight_acc)
        compute_accuracy = (compute_fitness, 1) # compute_accuracy is just compute_fitness with accuracy weight as 1

        # initialize whales and Leader (the agent with the max fitness)
        whales = initialize(self.num_agents, num_features)
        fitness = np.zeros(self.num_agents)
        accuracy = np.zeros(self.num_agents)
        Leader_agent = np.zeros((1, num_features))
        Leader_fitness = float("-inf")
        Leader_accuracy = float("-inf")

        # initialize convergence curves
        convergence_curve = {}
        convergence_curve['fitness'] = np.zeros(self.max_iter)

        # format the data 
        data = Data()
        val_size = float(input('Enter the percentage of data wanted for valdiation [0, 100]: '))/100
        data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(self.train_data, self.train_label, stratify=self.train_label, test_size=val_size)
        data_train = np.concatenate((data.train_X,data.train_Y.reshape(-1,1)), axis=1)
        data_val = np.concatenate((data.val_X,data.val_Y.reshape(-1,1)), axis=1)

        # create a solution object
        solution = Solution()
        solution.num_agents = self.num_agents
        solution.max_iter = self.max_iter
        solution.num_features = num_features
        solution.obj_function = self.obj_function

        # rank initial population
        whales, fitness = sort_agents(whales, obj, data)

        # start timer
        start_time = time.time()

        # main loop
        for iter_no in range(self.max_iter):
            print('\n================================================================================')
            print('                          Iteration - {}'.format(iter_no+1))
            print('================================================================================\n')

            a = 2 - iter_no * (2/self.max_iter)  # a decreases linearly fron 2 to 0
            # update the position of each whale
            for i in range(self.num_agents):
                # update the parameters
                r = np.random.random() # r is a random number in [0, 1]
                A = (2 * a * r) - a  # Eq. (3)
                C = 2 * r  # Eq. (4)
                l = -1 + (np.random.random() * 2)   # l is a random number in [-1, 1]
                p = np.random.random()  # p is a random number in [0, 1]
                b = 1  # defines shape of the spiral               
                
                if p<0.5:
                    # Shrinking Encircling mechanism
                    if abs(A)>=1:
                        rand_agent_index = np.random.randint(0, self.num_agents)
                        rand_agent = whales[rand_agent_index, :]
                        mod_dist_rand_agent = abs(C * rand_agent - whales[i,:]) 
                        whales[i,:] = rand_agent - (A * mod_dist_rand_agent)   # Eq. (9)
                        
                    else:
                        mod_dist_Leader = abs(C * Leader_agent - whales[i,:]) 
                        whales[i,:] = Leader_agent - (A * mod_dist_Leader)  # Eq. (2)
                    
                else:
                    # Spiral-Shaped Attack mechanism
                    dist_Leader = abs(Leader_agent - whales[i,:])
                    whales[i,:] = dist_Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_agent

                # Apply transformation function on the updated whale
                for j in range(num_features):
                    trans_value = trans_function(whales[i,j])
                    if (np.random.random() < trans_value): 
                        whales[i,j] = 1
                    else:
                        whales[i,j] = 0
            

            # update final information
            whales, fitness = sort_agents(whales, obj, data)
            
            #ABHC
            for i in range(whales.shape[0]):
                off = whales[i]
                whales[i] = AdaptiveBetaHC(off, fitness[i], self.classifier, data_train, data_val)
                
            display(whales, fitness, agent_name)
            if fitness[0]>Leader_fitness:
                Leader_agent = whales[0].copy()
                Leader_fitness = fitness[0].copy()

            convergence_curve['fitness'][iter_no] = np.mean(fitness)

            # convergence_curve['fitness'][iter_no] = np.mean(fitness)
            # convergence_curve['feature_count'][iter_no] = np.mean(np.sum(whales,axis=1))

        # compute final accuracy
        Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
        whales, accuracy = sort_agents(whales, compute_accuracy, data)

        print('\n================================================================================')
        print('                                    Final Result                                  ')
        print('================================================================================\n')
        print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
        print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
        print('Leader ' + agent_name + ' Classification Accuracy : {}'.format(Leader_accuracy))
        print('\n================================================================================\n')

        # stop timer
        end_time = time.time()
        exec_time = end_time - start_time
        
        # plot convergence graph
        fig, axes = Conv_plot(convergence_curve)
        if(self.save_conv_graph):
            plt.savefig('convergence_graph_'+ agent_name + '.jpg')
        plt.show()

        # update attributes of solution
        solution.best_agent = Leader_agent
        solution.best_fitness = Leader_fitness
        solution.best_accuracy = Leader_accuracy
        solution.convergence_curve = convergence_curve
        solution.final_population = whales
        solution.final_fitness = fitness
        solution.final_accuracy = accuracy
        solution.execution_time = exec_time

        return solution


    def AdaptiveBetaHC(offspring, offspring_fitness, classifier, arr_train, arr_val):
        
        b_min=1
        b_max = 100
        iters = 5
        for value in range(iters):
            neighbor = offspring
            percent = 0.3
            upper = int(percent*offspring.shape[0])
            if upper<=1:
                upper = offspring.shape[0]
            x = random.randint(1,upper)
            pos = random.sample(range(0, offspring.shape[0]-1),x)
            for i in pos:
                neighbor[i] = 1-neighbor[i]
            beta = b_min + (value/iters)*(b_max-b_min)
            for i in range(offspring.shape[0]):
                random.seed(0)
                if random.random() <= beta:
                    neighbor[i] = offspring[i]
            neighbor_fitness = eval_pop_fitness(np.asarray([neighbor]), classifier, arr_train, arr_val)
            if neighbor_fitness[0] > offspring_fitness:
                offspring = neighbor


        return offspring   



    def classifier_acc(val_pred, labels):
        
        count_list = np.where(val_pred==labels)
        acc = count_list[0].shape[0]/val_pred.shape[0]
        return acc


    def reduced_features(sol, arr_train, arr_val):
        
        sol = np.append(sol,[1])
        indices = np.where(sol==1)[0]
        train_space = arr_train[:,indices]
        val_space = arr_val[:,indices]
        return train_space, val_space


    def eval_pop_fitness(initial_parents, classifier, arr_train, arr_val):
            
            accuracies = np.zeros(initial_parents.shape[0])
            id=0
            for sol in initial_parents:
                train_space, val_space = reduced_features(sol, arr_train, arr_val)
                if classifier == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=5)
                    model.fit(train_space[:,:-1], train_space[:,-1])
                    val_pred = model.predict(val_space[:,:-1])
                    accuracies[id] = classifier_acc(val_pred, val_space[:,-1])
                    id +=1
            return accuracies

    
    def test(self, solution):

        # binary repr of feature space selection
        fspace_binary = solution.best_agent

        #indices of selected features
        feature_index = np.where(fspace_binary==1)[0]

        # refined feature space
        train_refined = self.train_data[:, feature_index]
        test_refined = self.test_data[:, feature_index]

        # auxiliary classifier
        
        model = RandomForestClassifier(max_depth=7, random_state=0)
        
        model.fit(train_refined, self.train_label)
        test_pred = model.predict(test_refined)

        #metrics
        print("Confusion Matrix:\n {} \n Classification Report:\n {} \n".format(confusion_matrix(test_pred, self.test_data), classification_report(test_pred, self.test_data)))
        
        acc = accuracy_score(test_pred, self.test_data)
        print("The accuracy on test dataset: {:.5f}%".format(acc*100))
