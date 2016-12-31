import theano
import theano.tensor as T
import lasagne
from customLayers import improvisedLayers, newLayers

# hyper-parameters as reported in the paper
embeddingDimension = 48
vocabSize = 15448
numOfLayers = 2
borderMode = 'full'
kTop = 4
numOfFilters1 = 6
numOfFilters2 = 14
filterSize1 = 7
filterSize2 = 5

# my own adaptations
batchSize = 1

# function to stack the layers and build the DCNN
def buildDCNN():
    
    layerInput = lasagne.layers.InputLayer(shape = (batchSize, None))
    
    layerEmbeddings = improvisedLayers.sentenceEmbeddingLayer(layerInput, 
                            vocabSize=vocabSize, embeddingDimension=embeddingDimension)
    
    layer1Convolutions = improvisedLayers.convolution1dLayer(layerEmbeddings, 
                            numOfFilters1, filterSize1, borderMode=borderMode)
    
    layer1Folding = newLayers.foldingLayer(layer1Convolutions)
    
    layer1dynamicKMaxPoolingLayer = newLayers.dynamicKMaxPoolingLayer(layer1Folding, 
                                        kTop = kTop, numOfLayers=numOfLayers, layerNumber = 1)
    
    layer1NonLinearity = lasagne.layers.NonlinearityLayer(layer1dynamicKMaxPoolingLayer, lasagne.nonlinearities.tanh)
    
    layer2Convolutions = improvisedLayers.convolution1dLayer(layer1NonLinearity, 
                            numOfFilters2, filterSize2, borderMode=borderMode)
    
    layer2Folding = newLayers.foldingLayer(layer2Convolutions)
    
    layerKTopPoolingLayer = newLayers.kTopPoolingLayer(layer2Folding, kTop = kTop)
    
    layer2NonLinearity = lasagne.layers.NonlinearityLayer(layerKTopPoolingLayer, lasagne.nonlinearities.tanh)
    
    layerDropout = lasagne.layers.DropoutLayer(layer2NonLinearity, p = 0.5) # p denotes dropout probability

    layerSoftmax = lasagne.layers.DenseLayer(layerDropout, 
                        num_units = 2, nonlinearity=lasagne.nonlinearities.softmax)

    return layerSoftmax