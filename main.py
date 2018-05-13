import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report


def genTrainData(dataSize):
    ################################################
    
    #    params:
    #        dataSize: generate trainData size      
    #    
    #    return: 
    #        train_X: input data
    #        train_Y: output data
    
    ################################################
    
    train_X = []
    train_Y = []
    tmp = []

    for i in range(int(dataSize/2)):
        a = rd.randint(0, 999)
        b = rd.randint(0, 999)
        str_a = str(a).zfill(3)
        str_b = str(b).zfill(3)
        list_of_a = [int(i) for i in str_a]
        list_of_b = [int(i) for i in str_b]

        tmp = list_of_a + list_of_b
        tmp.append(0)
        train_X.append(tmp)

        ans = a + b
        str_ans = str(ans).zfill(4)
        list_of_ans = [int(i) for i in str_ans]
        list_of_ans.append(0)
        train_Y.append(list_of_ans)

    for i in range(int(dataSize/2)): 
        a = rd.randint(0, 999)
        b = rd.randint(0, 999)
        str_a = str(a).zfill(3)
        str_b = str(b).zfill(3)
        list_of_a = [int(i) for i in str_a]
        list_of_b = [int(i) for i in str_b]

        tmp = list_of_a + list_of_b
        tmp.append(1)
        train_X.append(tmp)

        ans = a - b
        str_ans = str(abs(ans)).zfill(4)
        list_of_ans = [int(i) for i in str_ans]
        if ans < 0:
            list_of_ans.append(1)
        else:
            list_of_ans.append(0)
        train_Y.append(list_of_ans)
    
    pd.DataFrame(train_X, columns = ['num1(100)', 'num1(10)', 'num1(1)', 'num2(100)', 'num2(10)', 'num2(1)', '+/-']).to_csv("./data/train_X.csv", sep=',')
    pd.DataFrame(train_Y, columns = ['ans(1000)', 'ans(100)', 'ans(10)', 'ans(1)', '+/-']).to_csv("./data/train_Y.csv", sep=',')

def loadTrainData():
    train_X = pd.read_csv("./data/train_X.csv")
    train_Y = pd.read_csv("./data/train_Y.csv")
    
    X = train_X.loc[:, train_X.columns[1:]].values.tolist()
    Y = train_Y.loc[:, train_Y.columns[1:]].values.tolist()
    
    return X, Y

def encoding(data):    
    ################################################
    
    #    params:
    #        data: original input data      
    #    
    #    return: 
    #        encoded: data after one-hot encoding 
    
    ################################################
    
    encoded = to_categorical(data)
    return encoded

def flatten(data):    
    ################################################
    
    #    params:
    #        data: original input data      
    #    
    #    return: 
    #        encoded: data after flattening
    
    ################################################
    
    flatten = data.reshape(len(data), -1)
    return flatten

def genTestData(dataSize):
    test_X = []
    test_Y = []
    tmp = []

    for i in range(int(dataSize/2)):
        a = rd.randint(0, 999)
        b = rd.randint(0, 999)
        str_a = str(a).zfill(3)
        str_b = str(b).zfill(3)
        list_of_a = [int(i) for i in str_a]
        list_of_b = [int(i) for i in str_b]

        tmp = list_of_a + list_of_b
        tmp.append(0)
        test_X.append(tmp)

        ans = a + b
        str_ans = str(ans).zfill(4)
        list_of_ans = [int(i) for i in str_ans]
        list_of_ans.append(0)
        test_Y.append(list_of_ans)

    for i in range(int(dataSize/2)):
        a = rd.randint(0, 999)
        b = rd.randint(0, 999)
        str_a = str(a).zfill(3)
        str_b = str(b).zfill(3)
        list_of_a = [int(i) for i in str_a]
        list_of_b = [int(i) for i in str_b]

        tmp = list_of_a + list_of_b
        tmp.append(1)
        test_X.append(tmp)

        ans = a - b
        str_ans = str(abs(ans)).zfill(4)
        list_of_ans = [int(i) for i in str_ans]
        if ans < 0:
            list_of_ans.append(1)
        else:
            list_of_ans.append(0)
        test_Y.append(list_of_ans)
    
    pd.DataFrame(test_X, columns = ['num1(100)', 'num1(10)', 'num1(1)', 'num2(100)', 'num2(10)', 'num2(1)', '+/-']).to_csv("./data/test_X.csv", sep=',')
    pd.DataFrame(test_Y, columns = ['ans(1000)', 'ans(100)', 'ans(10)', 'ans(1)', '+/-']).to_csv("./data/test_Y.csv", sep=',')

def loadTestData():
    train_X = pd.read_csv("./data/test_X.csv")
    train_Y = pd.read_csv("./data/test_Y.csv")
    
    X = train_X.loc[:, train_X.columns[1:]].values.tolist()
    Y = train_Y.loc[:, train_Y.columns[1:]].values.tolist()
    
    return X, Y

def modelConstruction(train_X, train_Y):
    ################################################
    
    #    params:
    #        train_X: model training input      
    #        train_Y: model training output
    
    ################################################
    
    train_Y_thou = train_Y[:, 0, :]
    model_thou = Sequential()
    model_thou.add(Dense(70 , input_shape=(70,), activation='relu'))
    model_thou.add(Dense(35, activation='relu'))
    model_thou.add(Dense(10, activation='sigmoid'))
    model_thou.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("train model of thousands digit ...")
    model_thou.fit(train_X, train_Y_thou, batch_size=128, validation_split=0.2, shuffle=True, verbose=1, epochs=75)
    
    train_Y_hun = train_Y[:, 1, :]
    model_hun = Sequential()
    model_hun.add(Dense(250 , input_shape=(70,), activation='relu'))
    model_hun.add(Dense(250, activation='relu'))
    model_hun.add(Dense(150, activation='relu'))
    model_hun.add(Dense(50, activation='relu'))
    model_hun.add(Dense(10, activation='sigmoid'))
    model_hun.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("train model of hundreds digit ...")
    model_hun.fit(train_X, train_Y_hun, batch_size=128, validation_split=0.2, shuffle=True, verbose=1, epochs=100)
    
    train_Y_ten = train_Y[:, 2, :]
    model_ten = Sequential()
    model_ten.add(Dense(250 , input_shape=(70,), activation='relu'))
    model_ten.add(Dense(250, activation='relu'))
    model_ten.add(Dense(150, activation='relu'))
    model_ten.add(Dense(50, activation='relu'))
    model_ten.add(Dense(10, activation='sigmoid'))
    model_ten.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("train model of tens digit ...")
    model_ten.fit(train_X, train_Y_ten, batch_size=128, validation_split=0.2, shuffle=True, verbose=1, epochs=100)
    
    train_Y_unit = train_Y[:, 3, :]
    model_unit = Sequential()
    model_unit.add(Dense(250 , input_shape=(70,), activation='relu'))
    model_unit.add(Dense(250, activation='relu'))
    model_unit.add(Dense(150, activation='relu'))
    model_unit.add(Dense(50, activation='relu'))
    model_unit.add(Dense(10, activation='sigmoid'))
    model_unit.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("train model of units digit ...")
    model_unit.fit(train_X, train_Y_unit, batch_size=128, validation_split=0.2, shuffle=True, verbose=1, epochs=100)
    
    train_Y_sym = train_Y[:, 4, :2]
    model_sym = Sequential()
    model_sym.add(Dense(250 , input_shape=(70,), activation='relu'))
    model_sym.add(Dense(250, activation='relu'))
    model_sym.add(Dense(150, activation='relu'))
    model_sym.add(Dense(50, activation='relu'))
    model_sym.add(Dense(2, activation='sigmoid'))
    model_sym.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("train model of plus-minus sign ...")
    model_sym.fit(train_X, train_Y_sym, batch_size=128, validation_split=0.2, shuffle=True, verbose=1, epochs=50)

    model_dir = "./model/"
    model_thou.model.save(model_dir + "thousand.h5")
    model_hun.model.save(model_dir + "hundred.h5")
    model_ten.model.save(model_dir + "ten.h5")
    model_unit.model.save(model_dir + "unit.h5")
    model_sym.model.save(model_dir + "symbol.h5")

def loadModel():
    model_dir = "./model/"
    
    model_thou = load_model(model_dir + "thousand.h5")
    model_hun = load_model(model_dir + "hundred.h5")
    model_ten = load_model(model_dir + "ten.h5")
    model_unit = load_model(model_dir + "unit.h5")
    model_sym = load_model(model_dir + "symbol.h5")
    
    return model_thou, model_hun, model_ten, model_unit, model_sym

def plot_confusion_matrix(model, X, y, name):
    y_pred = np.argmax(model.predict(X, verbose=0), axis = 1)
    plt.figure(figsize=(8, 6))
    plt.title(name + " validation result")
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)

def modelValidation(test_X, test_Y):
    model_thou, model_hun, model_ten, model_unit, model_sym = loadModel()
    
    plot_confusion_matrix(model_thou, np.array(test_X), np.array(test_Y)[:, 0], "thousands model")
    plot_confusion_matrix(model_hun, np.array(test_X), np.array(test_Y)[:, 1], "hundreds model")
    plot_confusion_matrix(model_ten, np.array(test_X), np.array(test_Y)[:, 2], "tens model")
    plot_confusion_matrix(model_unit, np.array(test_X), np.array(test_Y)[:, 3], "units model")
    plot_confusion_matrix(model_sym, np.array(test_X), np.array(test_Y)[:, 4], "plus-minus sign model")
    
def dataRepresentation(test):    
    model_thou, model_hun, model_ten, model_unit, model_sym = loadModel()
  
    rd.shuffle(test)
    inputs = []
    actual = []
    for it in test:
        int1 = int(str(it[0]) + str(it[1]) + str(it[2]))
        int2 = int(str(it[3]) + str(it[4]) + str(it[5]))
        if it[6] == 0:
            symbol = '+'
            actual.append(int1 + int2)
        else:
            symbol = '-'
            actual.append(int1 - int2)

        input = str(int1) + symbol + str(int2)
        inputs.append(input)
    
    test_X = to_categorical(test).reshape(len(test), -1)
    thou_pred = np.argmax(model_thou.predict(test_X, verbose=0), axis = 1)
    hun_pred = np.argmax(model_hun.predict(test_X, verbose=0), axis = 1)
    ten_pred = np.argmax(model_ten.predict(test_X, verbose=0), axis = 1)
    unit_pred = np.argmax(model_unit.predict(test_X, verbose=0), axis = 1)
    sym_pred = np.argmax(model_sym.predict(test_X, verbose=0), axis = 1)

    predict = []
    for i in range(len(sym_pred)):
        if sym_pred[i] == 0:
            ans = 1
        else:
            ans = -1

        ans *= int(str(thou_pred[i]) + str(hun_pred[i]) + str(ten_pred[i]) + str(unit_pred[i]))
        predict.append(ans)
        
    correct = 0
    err = 0
    for i in range(len(actual)):
        if(actual[i] != predict[i]):
            err += 1
        else:
            correct += 1
    
    for i in range(10):
        if(actual[i] != predict[i]):
            print("%10s  \t [x]%6d" % (inputs[i], predict[i]))
        else:
            print("%10s  \t [o]%6d" % (inputs[i], predict[i]))
    
    accuracy = float(correct) / float(len(test)) * 100
    print ("Accuracy : %.2f %%" % accuracy)
    print ("Error times : %d / %d" % (err, len(test_X)))



if __name__ == '__main__':
	# Get the training data
	train_X, train_Y = loadTrainData()
	train = train_X
	train_X = flatten(encoding(train_X))
	train_Y = encoding(train_Y)

	# Train models
	# modelConstruction(train_X, train_Y)

	# Get the testing data
	test_X, test_Y = loadTestData()
	test = test_X
	test_X = flatten(encoding(test_X))

	# Validate models
	# modelValidation(test_X, test_Y)

	# Calculate the accuract on testing data
	dataRepresentation(train)
	print("========= Training Data ==========")

	print("")
	dataRepresentation(test)
	print("========= Validation Data =========")
