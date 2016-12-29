import theano.tensor as T
from lasagne.layers import Layer

class foldingLayer(Layer):

    def __init__(self, incoming, **kwargs):
        super(foldingLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input):
        # As per Kalchbrenner's paper, every two consecutive rows
        # in the output from the preceding convolutional layer
        # must be summed into one, so that the output from this 
        # 'folding' layer contains half the number of rows
        output = []
        # Input to this layer is a 4d tensor, and the number 
        # of rows in the input is the third dimension 
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for k in range(0, input_shape[2], 2):
                    outputRow = input[i, j, k:k+2, :].sum(axis = 0)
                    output.append(outputRow)

        output = T.reshape(T.concatenate(output), (self.get_output_shape_for(self.input_shape)))
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]/2, input_shape[3])


class dynamicKMaxPoolingLayer(Layer):

    def __init__(self, incoming, kTop, numOfLayers, layerNumber, **kwargs):
        super(dynamicKMaxPoolingLayer, self).__init__(incoming, **kwargs)

    self.kTop = kTop
    self.numOfLayers = numOfLayers
    self.layerNumber = layerNumber
    # As per the definition in Kalchbrenner's paper, the 
    # k value for k-max pooling is dynamically given as :
    self.k = T.cast(T.max([self.kTop, 
        T.ceil((self.numOfLayers - self.layerNumber)*self.input_shape[3]/float(self.numOfLayers))]), 'int16')

    def get_output_for(self, input)
        
        sortedIndices = T.argsort(input, axis = 3)
        kMaxIndices = sortedIndices[:, :, :, -self.k:]
        orderedKMaxIndices = T.sort(kMaxIndices)
        # create indices to the k-max elements to 
        # extract by flattening the input array 
        dim0 = T.arange(0,self.input_shape[0]).repeat(self.input_shape[1]*self.input_shape[2]*self.k)
        dim1 = T.arange(0,self.input_shape[1]).repeat(self.input_shape[2]*self.k).reshape((1,-1)).repeat(self.input_shape[0],axis=0).flatten()
        dim2 = T.arange(0,self.input_shape[2]).repeat(self.k).reshape((1,-1)).repeat(self.input_shape[0]*self.input_shape[1],axis=0).flatten()
        dim3 = orderedKMaxIndices.flatten()

        return input[dim0,dim1,dim2,dim3].reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2], self.k))

    def get_output_shape_for(self, input_shape):

        return input_shape[:2] + (self.k,)