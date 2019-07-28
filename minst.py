import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np

class MINST_NN():
    
    def __init__(self, train_data, test_data):
        
        self.train_data_path = train_data
        self.test_data_path = test_data
        pass
    
    #we start the NN by loading data and running the NN
    def start(self):
        ((train_input, test_input), (train_output, test_output)) = self.prepareData()
        self.startNN(train_input, train_output, test_input, test_output)
        pass
    
    # reads in the *.csv files and returning a tuple of tuples with pixel data (input) and 
    # the labels/classes (output)
    def prepareData(self):
        print("reading Data")
        all_train_values = np.array(pd.read_csv(self.train_data_path, header= None))
        all_test_values = np.array(pd.read_csv(self.test_data_path, header= None))
        
        print("preparing train data")
        (train_input, test_input) = self.prepareInputData(all_train_values, all_test_values)
        print("preparing test data")
        (train_output, test_output) = self.prepareOutputData(all_train_values, all_test_values)
        
        all_train_values = None
        all_test_values = None        
        
        return ((train_input, test_input), (train_output, test_output))
   
    
    # invoked in  prepareData(). Extract pixel values from *.csv file
    def prepareInputData(self, all_train_values, all_test_values):
        
        train_input = (np.array(all_train_values[:,1:], dtype = np.float32) / 255 * 0.99) + 0.01 
        test_input = (np.array(all_test_values[:,1:], dtype = np.float32) / 255 * 0.99) + 0.01        
        return (train_input, test_input)


    #invoked in prepareData(). Extract labels/classes/output from *csv file
    def prepareOutputData(self, all_train_values, all_test_values):
        
        test_output = np.empty((0,10), dtype = np.float32)
        train_output = np.empty((0,10), dtype = np.float32)
        
        for sample in all_test_values[:,0]:
            temp = np.zeros((1,10), dtype = np.float32)
            temp[0][sample] = 0.99
            test_output = np.append(test_output, temp, axis = 0)
            
        for sample in all_train_values[:,0]:
            temp = np.zeros((1,10), dtype = np.float32)
            temp[0][sample] = 0.99
            train_output = np.append(train_output, temp, axis = 0)
        
        
        return(train_output, test_output)


    #initializes and runns NN. Parameters are inside as local variables
    # for more on parameters, go to https://keras.io/
    def startNN(self, train_input, train_output, test_input, test_output):
        
        #needed parameters for NN
        EPOCH = 10
        BATCH = 10
        activation_sig = 'sigmoid'
        loss_function = 'categorical_crossentropy'
        optimizer = "adam"
        metrics = ["accuracy"]
        shuff = True
        input_layer = len(test_input[0])
        
        
        print("starting NN")
        model = Sequential()
        model.add(Dense(200, input_dim= input_layer, activation= activation_sig))
        model.add(Dense(10, activation=activation_sig))


        model.compile(loss=loss_function, optimizer= optimizer, metrics= metrics)
        history = model.fit(train_input, train_output, epochs=EPOCH, shuffle= shuff, batch_size= BATCH)


        loss , accuracy = model.evaluate(test_input, test_output)
        print('Accuracy: %.2f' % (accuracy*100))


        N = np.arange(0, EPOCH)
        pyplot.plot(N, history.history['acc'])
        pyplot.show()
        
        #self.predict(model, test_input, test_output)
        
    
    # we compare the first 1000 predictions. Not necessary and changeable for any other purpose
    def predict(self, model, test_input, test_output):
        predictions = model.predict_classes(test_input)
        for i in range(1000):
            for j in range(len(test_output[0])):
                if test_output[i][j] > 0.5:
                    print(predictions[i], j)
    
    
    pass



if __name__ == "__main__":
    
    # path to the files. If script and files are in the same direction, than path == filename
    X = MINST_NN("mnist_train.csv", "mnist_test.csv")
    X.start()








