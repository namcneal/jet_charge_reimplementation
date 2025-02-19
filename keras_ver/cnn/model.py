import os, sys

higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

##
from torch.utils.data import DataLoader
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from FileSystemNavigation import Directories, Filenames
from roc_sic_tools import down_quark_efficiency_roc

class CNNSpecification(object):
    def __init__(self, 
                    image_size, 
                    num_image_channels, 

                    num_conv_layers, 
                    conv_layer_num_filters, 
                    conv_layer_filter_sizes, 
                    conv_pooling_kernel_sizes, 
                    conv_dropout_percents, 

                    intermediate_dense_size, 
                    intermediate_dense_dropout, 
                    num_intermediate_dense_activation, 

                    num_final_dense_logits, 
                    final_dense_logits_activation, 

                    loss, opt, lr, decay):

        self.image_size = image_size
        self.num_image_channels = num_image_channels

        self.num_conv_layers = num_conv_layers
        self.conv_layer_num_filters = conv_layer_num_filters
        self.conv_layer_filter_sizes = conv_layer_filter_sizes
        self.conv_pooling_kernel_sizes = conv_pooling_kernel_sizes
        self.conv_dropout_percents = conv_dropout_percents

        self.intermediate_dense_size = intermediate_dense_size
        self.intermediate_dense_dropout = intermediate_dense_dropout
        self.num_intermediate_dense_activation = num_intermediate_dense_activation

        self.num_final_dense_logits = num_final_dense_logits
        self.final_dense_logits_activation = final_dense_logits_activation

        self.loss = loss
        self.opt = opt
        self.lr = lr

        self.decay = decay

    @staticmethod
    def default():
        return CNNSpecification(
            image_size=33, 
            num_image_channels=2,

            num_conv_layers=3,
            conv_layer_num_filters=[64, 64, 64],
            conv_layer_filter_sizes=[8, 4, 4],
            conv_pooling_kernel_sizes=[2, 2, 2],
            conv_dropout_percents=[0.18, 0.35, 0.35],

            intermediate_dense_size=64,
            intermediate_dense_dropout=0.35,
            num_intermediate_dense_activation='silu',

            num_final_dense_logits=2,
            final_dense_logits_activation='softmax',

            loss='categorical_crossentropy',
            opt=Adam,
            lr=0.001,
            decay=0.0
        )

class CNN(object):
    def __init__(self, spec:CNNSpecification):
        self.specification = spec
        self.model = None

    def create_model(self):
        model = Sequential()

        params_each_layer = zip(self.specification.conv_layer_num_filters, 
                                self.specification.conv_layer_filter_sizes, 
                                self.specification.conv_pooling_kernel_sizes,
                                self.specification.conv_dropout_percents)

        for i,(num_filters, filter_size, pooling_kernel_size, dropout_percent) in enumerate(params_each_layer):
            kwargs : Dict = {}
            is_input_layer = i > 0
            if is_input_layer:
                kwargs = {'input_shape': (self.specification.num_image_channels, self.specification.image_size, self.specification.image_size)}

            model.add(Conv2D(num_filters, filter_size, 
                                kernel_initializer = 'he_uniform', 
                                padding = 'valid',
                                activation = act,
                                **kwargs)) 
            model.add(MaxPooling2D(pool_size = pooling_kernel_size))
            model.add(dropout_layer(dropout_percent))

        model.add(Flatten())

        model.add(Dense(self.specification.intermediate_dense_size, activation = self.specification.num_intermediate_dense_activation))
        model.add(Dropout(self.specification.intermediate_dense_dropout))

        model.add(Dense(self.specification.num_final_dense_logits, activation = self.specification.final_dense_logits_activation))

        if comp:
            model.compile(loss = self.specification.loss, optimizer = self.specification.opt(lr = self.specification.lr, decay = self.specification.decay), metrics = ['accuracy'])
            if summary:
                model.summary()

        self.model = model

    def train_model(self, image_label_data:torch.Dataloader, validation_pct, batch_size, epochs):
        history = self.model.fit(
            x                 = image_label_data,
            batch_size        = batch_size, 
            epochs            = epochs, 
            validation_split  = validation_pct,
            verbose           = 1,
            shuffle           = True,
        )

        return history


    def save_model(self, model_filename_w_format:str, save_directory:str):
        filename = model_filename_w_format

        if dir[-1] != '/':
            dir += '/'

        model.save(save_directory + filename)

    def evaluate_model(self, directories:Directories, filenames:Filenames,
                        image_dataloader:torch.DataLoader,
                        labels:np.ndarray):

        probability_predictions = model.predict(image_dataloader)

        plot_dir  = directories.save_data_directory
        plot_name = filenames.roc_curve_filename(filenames.kappa, filenames.energy_gev)

        down_quark_efficiency_roc(probability_predictions, labels, plot_dir, plot_name)


