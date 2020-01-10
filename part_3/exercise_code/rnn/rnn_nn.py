import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        """self.layer = nn.Linear(input_size, hidden_size)
        self.hidden = hidden_size
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()"""
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.i2o = nn.Linear(input_size, 1)
        self.act = nn.Tanh() 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        """
        last = []
        for row in x:
            print("original: ", row)
            row = self.layer(row)
            print("layer: ", row)
            if len(h_seq) > 0:
                row += h_seq[-1]
            row = self.activation(row)
            print("activated: ", row)
            print()
            h_seq.append(row)
        last.append(row)
        
        h = torch.stack(last)
        h_seq = torch.stack(h_seq)
        
        last = []
        for i in range(self.hidden):
            x = self.layer(x)
            if len(h_seq) > 0:
                x += h_seq[-1]
            x = self.activation(x)
            h_seq.append(x)
        last.append(x[-1])
        
        h = torch.stack(last)
        h_seq = h_seq[-1] """

        hidden = self.i2h(x)
        output = self.i2o(x)
        output = self.act(output)
        h_seq = hidden
        h = output
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h
    
    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
    ############################################################################
    # TODO: Build a one layer LSTM with an activation with the attributes      #
    # defined above and a forward function below. Use the nn.Linear() function #
    # as your linear layers.                                                   #
    # Initialse h and c as 0 if these values are not given.                    #
    ############################################################################
        pass
       
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        pass
       
    def forward(self, x):
        pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a LSTM classifier                                           #
    ############################################################################
        pass
    
    def forward(self, x):
        pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
