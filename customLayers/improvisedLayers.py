from lasagne.layers import Layer, EmbeddingLayer


class sentenceEmbeddingLayer(EmbeddingLayer):

    def __init__(self, incoming, vocabSize, embeddingDimension, **kwargs):
        super(sentenceEmbeddingLayer, self).__init__(incoming, input_size = vocabSize, 
            output_size = embeddingDimension, **kwargs)

        # One 'zero' embedding needs to be present to account for the out-of-vocabulary words
        self.W = T.concatenate(self.W, T.zeros((1, embeddingDimension)))

    # FUNCTIONS TO BE OVERRIDDEN FOR THE ARCHITECTURE OF DCNN
    def get_output_for(self, input):
        # For a sentence in a batch, the sentence matrix should be passed to the covolutional
        # layers with dimensions 'd' x 'm', as described in Kalchbrenner's paper .
        return T.transpose(self.W[input], (0,2,1))

    # self.input_shape is a 2-tuple
    # self.input_shape[0] denotes batch size
    # self.input_shape[1] denotes sentences length (uniform throughout the batch)
    def get_output_shape_for(self, input):
        return (self.input_shape[0], self.output_size, self.input_shape[1])


class convolution1dLayer(Layer):
    """
    PARAMETERS :-
    
    numFilters :    number of different filters to apply
    filterSize :    size of each of the filters 
    border_mode :   'full' for wide convolutions as described in Kalchbrenner's paper.
                    'valid' for narrow convolutions.
    """
    def __init__(self, incoming, numOfFilters, filterSize, 
        W = lasagne.init.GlorotUniform(), b = lasagne.init.constant(0), 
        border_mode = 'full', **kwargs):
        super(convolutions1d, self).__init__(incoming, **kwargs)

    # Shape of the input:
    # If it is the first convolutional layer, i.e. just after the sentenceEmbeddingLayer,
    # then the input shape will be 3(batch size, embeddingDimension, length of sentences).
    # If it is not the first convolutional layer, then input shape 
    # will be 4(batch size, no of input channels, embeddingDimesion, length of sentences ) 
    if len(self.input_shape) == 3:
        self.numOfInputChannels = 1
        self.numOfRows = self.input_shape[1]
    else:
        self.numOfInputChannels = self.input_shape[1]
        self.numOfRows = self.input_shape[2]

    self.numOfFilters = numOfFilters
    self.filterSize = filterSize
    self.borderMode = border_mode
    self.nonLinearity = nonLinearity

    # To register the parameters for this layer
    # 1. W : The filters for the convolutions
    # 2. b : The bias values for each filter   
    self.W = self.add_param(W, shape = (self.numOfFilters, self.numOfInputChannels, self.numOfRows, self.filterSize))
    if b:
        self.b = self.add_param(b, shape = (self.numOfFilters,))
    else:
        self.b = None

    def get_output_for(self, input):

        # Theano does not support 1d convolutions, so to convolve row wise on the sentence matrix,
        # each row will be passed one by one to the tensor.nnet.conv2d function of theano and the 
        # corresponding row of a filter will convolve over this row of the sentence matrix

        # The shape of the input to the T.nnet.conv2d function needs to be provided as an argument
        if len(self.input_shape) == 3: # if this is the first convolution1dLayer layer
            # self.input_shape[0] represents batch size.
            inputShape = (self.input_shape[0], self.numOfInputChannels, 1, self.input_shape[2])
        else:
            inputShape = (self.input_shape[0], self.numOfInputChannels, 1, self.input_shape[3])
        
        # The shape of the filter needs to be provided as an argument to the T.nnet.conv2d function
        filterShape = (self.numOfFilters, self.numOfInputChannels, 1, self.filterSize)

        convolutions = []

        for i in xrange(self.numOfRows):
            convolutions.append(
                T.nnet.conv2d(input[:,:,i,:].dimshuffle(0,1,'x',2), 
                    self.W[:,:,i,:].dimshuffle(0,1,'x',2),
                    input_shape = inputShape,
                    filter_shape = filterShape,
                    border_mode = self.borderMode
                    subsample = (1,1)
                )
            )

        convolutions = T.concatenate(convolutions, axis = 2)
        
        if self.b:
            convolutions = convolutions + self.b.dimshuffle('x', 0, 'x', 'x')
        
        return convolutions

    def get_output_shape_for(self, inputShape):
        
        if self.borderMode == 'full':
            numOfOutputCols = inputShape[-1]+self.filterSize-1
        elif self.borderMode == 'valid':
            numOfOutputCols = inputShape[-1]-self.filterSize+1

        outputShape = (inputShape[0], self.numOfFilters, self.numOfRows, numOfOutputCols)
        return outputShape