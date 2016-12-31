import theano
import theano.tensor as T
import lasagne
import architecture
import loader
import cPickle
import numpy as np
from lasagne.regularization import regularize_layer_params_weighted, l2

NUMOFEPOCHS = 1

output = architecture.buildDCNN()
dcnnParams = lasagne.layers.get_all_params(output)

# SYMBOLIC INPUTS
x = T.imatrix()
y = T.ivector()

# Without L2 Regularization 
loss = lasagne.objectives.aggregate(
    lasagne.objectives.categorical_crossentropy(
        lasagne.layers.get_output(output, x), y), mode = 'mean')
updates = lasagne.updates.adagrad(loss, dcnnParams, learning_rate = 0.1)

# ACCURACY FOR PREDICTIONS
prediction = T.argmax(lasagne.layers.get_output(output, x, deterministic=True), axis=1)
score = T.eq(prediction, y).mean()

# SYMBOLIC FUNCTIONS
trainDCNN = theano.function([x,y], outputs = loss, updates = updates)
validateDCNN = theano.function([x,y], outputs = score)
testDCNN = theano.function([x,y], outputs = score)

# LOAD THE DATA
trainingSentences = loader.loadData('myDataset/train.txt')
trainingLabels = loader.loadData('myDataset/train_label.txt')
validationSentences = loader.loadData('myDataset/dev.txt')
validationLabels = loader.loadData('myDataset/dev_label.txt')
testSentences = loader.loadData('myDataset/test.txt')
testLabels = loader.loadData('myDataset/test_label.txt')

# TRAIN THE MODEL
print '...training the DCNN'
for epoch in range(NUMOFEPOCHS):
    for i in xrange(len(trainingSentences)):
        trainDCNN(np.asarray(trainingSentences[i:i+1], dtype = np.int32), 
            np.asarray(trainingLabels[i], dtype = np.int32))
        print 'Sentence ', i, ' complete.'

# SAVE THE TRAINED MODEL
parameters = lasagne.layers.get_all_param_values(output)
with open('DCNNParameters.pkl', 'wb') as file:
    cPickle.dump(parameters, file, protocol = 2)

# VALIDATE THE MODEL
print '...running the DCNN on Validation Set'

accuracy = 0
for i in xrange(len(validationSentences)):
    score = validateDCNN(np.asarray(validationSentences[i:i+1], dtype = np.int32), 
        np.asarray(validationLabels[i], dtype = np.int32))
    accuracy += score
    print 'Sentence ', i, ' complete.'

accuracy /= float(len(validationSentences))
print "Accuracy in Validation =", accuracy


# TEST THE MODEL
print '...running the DCNN on Test Set'

accuracy = 0
for i in xrange(len(testSentences)):
    score = validateDCNN(np.asarray(testSentences[i:i+1], dtype = np.int32), 
        np.asarray(testLabels[i], dtype = np.int32))
    accuracy += score
    print 'Sentence ', i, ' complete.'

accuracy /= float(len(testSentences))
print "Accuracy in Test =", accuracy