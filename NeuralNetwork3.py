import pandas as pda
import numpy as np
import csv
import sys
import time

def load_data(train_image_file, train_label_file, test_image_file):
    
    train_image = pda.read_csv(train_image_file, header=None).values
    test_image = pda.read_csv(test_image_file, header=None).values
    
    train_label = pda.read_csv(train_label_file, header=None)
    train_label_enc = pda.get_dummies(train_label[0]).values  
    
    return (train_image / 255), (test_image / 255), train_label_enc, train_label.values


def init_weights(layers):
    weight = {}

    for i in range(1, 4):
        weight['W' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * 0.01
        weight['b' + str(i)] = np.zeros((layers[i], 1))

    return weight


def forward_propagation(train_data, weight):
    local_cache = []
    A = train_data.T

    for i in range(1, 3):
        # linear formula
        Z = np.dot(weight["W" + str(i)], A) + weight["b" + str(i)]

        # sigmoid activation
        A = 1. / (1. + np.exp(-Z))
        # print(np.amax(weight["W" + str(i)]))

        local_cache.append((Z, A))

    # linear formula
    Z = np.dot(weight["W" + str(3)], A) + weight["b" + str(3)]

    # softmax activation
    exp = np.exp(Z - np.max(Z))
    A = exp / exp.sum(axis=0)

    local_cache.append((Z, A))
    return local_cache


def backward_propagation(train_data, train_label, forward_cache, weight):
    gradient = {}
    m = len(train_label)

    # cross entropy loss gradient for softmax
    # https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
    # grad_softmax = Ok - yk = softmax() - one-hot encoded ground truth
    # = activation cache of last layer - training label

    dL = forward_cache[2][1] - train_label.T

    # 4
    gradient["dA" + str(3)] = np.dot(dL, forward_cache[1][1].T) / m
    gradient["db" + str(3)] = np.sum(dL, axis=1, keepdims=True) / m

    dA_2 = np.dot(weight["W" + str(3)].T, dL)
    dZ_2 = forward_cache[1][1] * (1 - forward_cache[1][1])    

    # 2
    gradient["dA" + str(2)] = np.dot(dA_2 * dZ_2, forward_cache[0][1].T) / m
    gradient["db" + str(2)] = np.sum(dA_2 * dZ_2, axis=1, keepdims=True) / m

    dA_1 = np.dot(weight["W" + str(2)].T, dA_2)
    dZ_1 = forward_cache[0][1] * (1 - forward_cache[0][1])

    # 3
    gradient["dA" + str(1)] = np.dot(dA_1 * dZ_1, train_data) / m
    gradient["db" + str(1)] = np.sum(dA_1 * dZ_1, axis=1, keepdims=True) / m

    return gradient


def update_weights(weight, grads, learning_rate):
    weight["W" + str(3)] -= learning_rate * grads["dA3"]
    weight["b" + str(3)] -= learning_rate * grads["db3"]
    weight["W" + str(2)] -= learning_rate * grads["dA2"]
    weight["b" + str(2)] -= learning_rate * grads["db2"]
    weight["W" + str(1)] -= learning_rate * grads["dA1"]
    weight["b" + str(1)] -= learning_rate * grads["db1"]


if __name__ == '__main__':

    start = time.time()
    arg_1 = sys.argv[1]
    arg_2 = sys.argv[2]
    arg_3 = sys.argv[3]

    train_X, test_X, train_Y, train_Y_orig = load_data(arg_1, arg_2, arg_3)
    weights = init_weights([784, 512, 256, train_Y.shape[1]])
    alpha = 0.8    
    epoch = 50
    batch_size = 10
    step = (alpha - 0.01)/epoch

    for iteration in range(epoch):
        #print(iteration)
        for batch in range(0, len(train_X), batch_size):
            cache = forward_propagation(train_X[batch:batch + batch_size], weights)
            gradients = backward_propagation(
                train_X[batch:batch + batch_size],
                train_Y[batch:batch + batch_size],
                cache,
                weights)
            update_weights(weights, gradients, alpha)              
        alpha = alpha - (step)      
        
        
        #predictions = []
        #results = forward_propagation(train_X, weights)
        #preds = np.argmax(results[2][1], axis=0)
        #print(results[2][1])
        #for index in range(0, preds.shape[0]):
            #predictions.append(preds[index] == train_Y_orig[index])
        #print("Accuracy: " + str(np.mean(predictions)))
        
        
        
    results = forward_propagation(test_X, weights)
    predictions = np.argmax(results[2][1], axis=0)    
    np.savetxt("test_predictions.csv", predictions, fmt='% s')
    end = time.time()
    print(end - start)
