import torch 

class ConvolutionalLayer(torch.nn.Module):
    def __init__(self, num_channels, num_kernels, filter_size, dropout):
        self.conv = torch.nn.Conv2d(num_channels, num_kernels, kernel_size=filter_size)
        self.conv_activation = torch.nn.ReLU()
        self.max_pooling     = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_activation(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x
    
class PostConvDenseLayer(torch.nn.Module):
    def __init__(self, num_out, dropout=0.35):
        self.dense = torch.nn.LazyLinear(num_out)
        self.dense_activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dense(x)
        x = self.dense_activation(x)
        return x
    

class CNN(torch.nn.Module):
    def __init__(self, num_conv_channels):

        ## Parameters for the three convolutional layers
        num_channels = num_conv_channels
        num_filters  = 64
        filter_sizes = [8, 4, 4]
        dropouts     = [0.18, 0.35, 0.35]

        ## Creating all three of the layers
        self.conv_layers = [ConvolutionalLayer(num_channels, num_filters, filter_size, dropout) for filter_size, dropout in zip(filter_sizes, dropouts)]

        ## The dense layer after the convolutional layers
        num_dense_layer_neurons = 64
        self.dense = PostConvDenseLayer(num_dense_layer_neurons)

        ## The classifier layer
        num_classifier_neurons = 2
        self.classifier_neurons  = torch.nn.Linear(num_dense_layer_neurons, num_classifier_neurons)

        self.layers = torch.nn.Sequential(
                *self.conv_layers, 

                torch.nn.Flatten(),
                self.dense, 

                self.classifier_neurons,
                torch.nn.Sofrtmax(dim=1)
        )

    def forward(self, x):
        self.layers(x)

