'''
Created on Dec 19, 2017

@author: Anunay Amar
'''
import csv
import numpy as np

class LogisticRegressionClassification:
    def __init__(self):
        self.dataset = None
        self.identifier = []
        print ("In init")
    
    #Load the training csv
    def load_csv(self, file_location):
        data = []
        ignoreHeader = False
        with open(file_location, 'rt') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not ignoreHeader:
                    ignoreHeader = True
                    continue
                print (row)
                selected_features = self.feature_engineering(row)
                if selected_features:
                    data.append(selected_features)
                    self.identifier.append(row[0])

        total_data = len(data)
        print (total_data)
        self.dataset = np.array(data, dtype = 'float64')
        
        #It is very important to shuffle the data, to avoid any bias due to inherent ordering of the data
        np.random.shuffle(self.dataset)
        print (self.dataset.shape)

        x = self.dataset[:,:-1]
        y = self.dataset[:,-1].reshape(total_data,1)
        
        #Split training and testing set into 80-20 ratio
        train_size = (total_data*80)/100
        test_size = total_data - train_size
        
        x_train_set = x[:train_size,:]
        y_train_set = y[:train_size,:]
        
        x_validation_set = x[train_size:,:]
        y_validation_set = y[train_size:,:]
        
        return x_train_set, y_train_set, x_validation_set, y_validation_set
        
        
    #Loads the test csv    
    def load_test_csv(self, file_location):
        data = []
        self.identifier = []
        ignoreHeader = False
        with open(file_location, 'rt') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not ignoreHeader:
                    ignoreHeader = True
                    continue
                print (row)
                selected_features = self.feature_engineering_test(row)
                if selected_features:
                    data.append(selected_features)
                    self.identifier.append(row[0])

        total_data = len(data)
        print (total_data)
        self.dataset = np.array(data, dtype = 'float64')
        return total_data        


    #Initialize the weight and bias parameters
    def initialize_with_zeros(self, dim):
        w = np.zeros(shape=(dim, 1))
        b = 0    
        return w, b

   
    #Performs feature engineering and feature selection on training set
    def feature_engineering(self, row):
        if not (row[1] and row[2] and row[4] and row[5] and row[9]):
            return None       
        
        survived = int(row[1])
        passenger_class = int(row[2])
        sex = None
        if row[4] == "male":
            sex = 1
        else:
            sex = 2            
        age = row[5]
        siblings = 0
        parents = 0
        if row[6]:
            siblings = int(row[6])
        if row[7]:
            parents = int(row[7]) 
            
        family_size = siblings + parents + 1
        ticket_price = float(row[9]) / family_size
        embarked = row[11]
        
        if embarked == "Q":
            embarked = 1*1
        elif embarked == "C":
            embarked = 2*2
        else:
            embarked = 3*3
            
        
        data = [passenger_class*passenger_class, sex*sex, age, ticket_price, siblings, parents, embarked, survived]
        return data
    

    #Performs feature engineering and feature selection on test set
    def feature_engineering_test(self, row):
        passenger_class = int(row[1])
        sex = None
        if row[3] == "male":
            sex = 1
        else:
            sex = 2
        age = row[4]
        if not age:
            age=20
        siblings = 0
        parents = 0
        if row[5]:
            siblings = int(row[5])
        if row[6]:
            parents = int(row[6]) 
            
        family_size = siblings + parents + 1
        fare_paid = row[8]
        if fare_paid:
            fare_paid = float(fare_paid)
        else:
            if passenger_class > 0.5:
                fare_paid = 30.0
            elif passenger_class == 0.5:
                fare_paid = 15.0
            else:
                fare_paid = 7.9
            
        ticket_price = fare_paid / family_size
        
        embarked = row[10]
        if embarked == "Q":
            embarked = 1*1
        elif embarked == "C":
            embarked = 2*2
        else:
            embarked = 3*3
        
        
        data = [passenger_class*passenger_class, sex*sex, age, ticket_price, siblings, parents, embarked]
        return data
    
        
         
    
    def weighted_sum(self, x, w, b):        
        #Calculating the weighted sum
        z = np.dot(x, w) + b
        return z
    
    def activation_function(self, z):
        #Applying sigmoid activation function
        return 1/(1+np.exp(-1*z))
    
    
    #Implements gradient descent with L2 regularization
    def gradient_descent(self, x, w, b, a, y, learning_rate, lambda_parameter):
        #number of examples
        m = x.shape[0]
        
        #a - y, dim (m,1)
        #This is for the x0 which is always 1, the bias unit.
        b = b - learning_rate * np.sum(np.subtract(a, y))/m
        
        #This is without regularization. w is weight vector, dimension (n, 1)
        #w = w - learning_rate * ((np.sum(np.multiply(np.subtract(a, y), x), axis=0, keepdims=True))/m).T

        #This is with regularization. w is weight vector, dimension (n, 1)    
        w = w - learning_rate * (((np.sum(np.multiply(np.subtract(a, y), x), axis=0, keepdims=True))/m).T + (lambda_parameter/m)*w) 
        return w, b
        
    
    #This function implements the logistic regression.
    def logistic_regression(self, x, y, w, b, num_of_iteration, learning_rate, lambda_parameter):
        number_of_examples = x.shape[0]
        
        for i in range(0, num_of_iteration):
            #1. Performs weighted sum of input, weights and bias term
            #2. Applies sigmoid activation function to the weighted sum
            #3. Performs gradient descent
            z = self.weighted_sum(x, w, b)
            a = self.activation_function(z)            
            w, b = self.gradient_descent(x, w, b, a, y, learning_rate, lambda_parameter)
            
            loss = (np.sum(y*np.log(a) + (1-y)*np.log(1-a))/(-1*number_of_examples)) + (np.sum(np.square(w))*lambda_parameter)/(2*number_of_examples)
            print ("Iteration:" + str(i) + " Loss:" +str(loss))
        return w, b
    
    
    #This function calculates the accuracy on training and validation set
    def test(self, w, b, x_validation, y_validation, type):
        z = self.weighted_sum(x_validation, w, b)  
        a = self.activation_function(z)
        
        size_validation = x_validation.shape[0] 
        y_prediction = np.zeros((size_validation, 1))
        for i in range(size_validation):
            if a[i,0] > 0.5:
                y_prediction[i,0] = 1
        
        print (type + " Accuracy:" + str((100 - np.mean(np.abs(y_prediction - y_validation))*100)))
        
    
    #This function loads the test set, and generates the result csv
    def test_set_accuracy(self, file_location, w, b):
        total_data = self.load_test_csv(file_location)
        x = self.dataset

        z = self.weighted_sum(x, w, b)  
        a = self.activation_function(z)
        
        size_validation = x.shape[0] 
        y_prediction = np.zeros((size_validation, 1))
        
        fileWriter = open("result.csv", "w")
        fileWriter.write("PassengerId,Survived\n")
        for i in range(size_validation):
            if a[i,0] > 0.5:
                y_prediction[i,0] = 1
                fileWriter.write(str(self.identifier[i]) + "," + str(1) + "\n")
            else:
                fileWriter.write(str(self.identifier[i]) + "," + str(0) + "\n")
        fileWriter.close()
        
        
    
    def main(self, file_location, test_file_location):
        x_train_set, y_train_set, x_validation_set, y_validation_set = self.load_csv(file_location)
        
        dims = x_train_set.shape[1]        
        w, b = self.initialize_with_zeros(dims)
        w, b = self.logistic_regression(x_train_set, y_train_set, w, b, num_of_iteration=1300000, learning_rate=0.001, lambda_parameter = 10)
        self.test(w, b, x_train_set, y_train_set, type= "Train")
        self.test(w, b, x_validation_set, y_validation_set, type= "Validation")
        
        self.test_set_accuracy(test_file_location, w, b)
        
        
    
LogisticRegressionClassification().main("data/train.csv", "data/test.csv")