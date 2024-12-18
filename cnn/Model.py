import torch 

class ConvolutionalLayer(torch.nn.Module):
    def __init__(self, num_channels, num_kernels, filter_size, dropout):
        super().__init__()

        self.units  = torch.nn.Conv2d(num_channels, num_kernels, kernel_size=filter_size)
        self.weight = self.units.weight
        self.bias   = self.units.bias

        self.conv_activation = torch.nn.ReLU()
        self.max_pooling     = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.units(x)
        x = self.conv_activation(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x
    
class PostConvDenseLayer(torch.nn.Module):
    def __init__(self, num_out, dropout=0.35):
        super().__init__()

        self.units  = torch.nn.LazyLinear(num_out)
        self.weight = self.units.weight
        self.bias   = self.units.bias

        self.dense_activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.units(x)
        x = self.dense_activation(x)
        return x

class CNN(torch.nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()

        ## Parameters for the three convolutional layers
        num_filters  = 64
        in_channels  = [num_input_channels, num_filters, num_filters]
        out_channels = [num_filters,        num_filters, num_filters]
        # num_channels = num_conv_channels

        filter_sizes = [8, 4, 4]
        dropouts     = [0.18, 0.35, 0.35]

        ## Creating all three of the layers
        self.conv_layers = torch.nn.Sequential(
            *[ConvolutionalLayer(in_channels[i], out_channels[i], filter_sizes[i], dropouts[i]) for i in range(3)]
        )

        ## The dense layer after the convolutional layers
        num_dense_layer_neurons = 64
        self.dense = PostConvDenseLayer(num_dense_layer_neurons)

        ## The classifier layer
        num_classifier_neurons = 2
        self.classifier_logits  = torch.nn.Linear(num_dense_layer_neurons, num_classifier_neurons)

        self.layers = torch.nn.Sequential(
                *self.conv_layers, 
                self.dense, 
                self.classifier_logits,
        )

        ## Run the model on a dummy input to initialize the weights
        dummy_image = torch.zeros((1, num_input_channels, 33, 33))
        self.layers(dummy_image)

        # for layer in self.layers:
            # torch.nn.init.xavier_normal_(layer.weight)
            # torch.nn.init.normal_(layer.bias)

            # torch.nn.init.xavier_uniform_(layer.weight)
            # torch.nn.init.uniform_(layer.bias)

    def forward(self, x):
        return self.layers(x)

        # x = self.conv_layers(x)
        # x = torch.nn.Flatten()(x)
        # x = self.dense(x)
        # x = self.classifier_neurons(x)
        # x = torch.nn.Softmax(dim=1)(x)
        # return x

