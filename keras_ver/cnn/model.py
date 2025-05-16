import os, sys

higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

##
import numpy as np 

from torch.utils.data import DataLoader
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from JetImages import PreprocessingSpecification
from FileSystemNavigation import Directories, Filenames, DataDetails
from roc_sic_tools import down_quark_efficiency_roc_and_sic

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
                    intermediate_dense_activation, 

                    num_final_dense_logits, 
                    final_dense_logits_activation, 

                    stopping_patience,
                    loss, 
                    opt,):

        self.image_size = image_size
        self.num_image_channels = num_image_channels

        self.num_conv_layers = num_conv_layers
        self.conv_layer_num_filters    = conv_layer_num_filters
        self.conv_layer_filter_sizes   = conv_layer_filter_sizes
        self.conv_pooling_kernel_sizes = conv_pooling_kernel_sizes
        self.conv_dropout_percents     = conv_dropout_percents
        self.conv_activation           = intermediate_dense_activation

        self.intermediate_dense_size       = intermediate_dense_size
        self.intermediate_dense_dropout    = intermediate_dense_dropout
        self.intermediate_dense_activation = intermediate_dense_activation

        self.num_final_dense_logits        = num_final_dense_logits
        self.final_dense_logits_activation = final_dense_logits_activation

        self.stopping_patience = stopping_patience
        self.loss = loss
        self.opt = opt

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
            intermediate_dense_activation='relu',

            num_final_dense_logits=2,
            final_dense_logits_activation='softmax',

            stopping_patience=5,
            loss='categorical_crossentropy',
            opt=keras.optimizers.Adam(learning_rate=0.001),
        )

class CNN(object):
    def __init__(self, specification:CNNSpecification):
        self.specification = specification
        self.model = CNN.create_model(self.specification)        

    @staticmethod
    def create_model(specification:CNNSpecification, comp=True, summary=True):
        model = Sequential()

        params_each_layer = zip(specification.conv_layer_num_filters, 
                                specification.conv_layer_filter_sizes, 
                                specification.conv_pooling_kernel_sizes,
                                specification.conv_dropout_percents)

        for i,(num_filters, filter_size, pooling_kernel_size, dropout_percent) in enumerate(params_each_layer):
            kwargs : Dict = {}
            # is_input_layer = i > 0
            # if is_input_layer:
            #     kwargs = {'input_shape': (specification.num_image_channels, specification.image_size, specification.image_size)}

            model.add(Conv2D(num_filters, filter_size, 
                                kernel_initializer = 'he_uniform', 
                                padding = 'valid',
                                activation = specification.conv_activation,
                                data_format = 'channels_first',
                                **kwargs)) 
            model.add(MaxPooling2D(pool_size = pooling_kernel_size))
            model.add(Dropout(dropout_percent))

        model.add(Flatten())

        model.add(Dense(specification.intermediate_dense_size, activation = specification.intermediate_dense_activation))
        model.add(Dropout(specification.intermediate_dense_dropout))

        model.add(Dense(specification.num_final_dense_logits, activation = specification.final_dense_logits_activation))

        if comp:
            model.compile(loss = specification.loss, optimizer = specification.opt, metrics = ['accuracy'])
            if summary:
                model.summary()

        return model

    def train(self, directories:Directories, filenames:Filenames, 
                jet_charge_kappa:float, preprocessing_details:str,
                training_image_label_data:DataLoader, 
                validation_image_label_data:DataLoader,
                batch_size:int, epochs:int):

        checkpoint_filename  = "checkpoint_" + filenames.saved_model_filename(jet_charge_kappa, preprocessing_details)
        checkpoint_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        checkpoint_filepath  = os.path.join(checkpoint_directory, checkpoint_filename)  

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=self.specification.stopping_patience,
            mode="auto",
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_freq="epoch",
            monitor='val_loss',
            mode='max',
            save_best_only=True
        )
        
        history = self.model.fit(
            x                 = training_image_label_data,
            validation_data   = validation_image_label_data,
            batch_size        = batch_size, 
            epochs            = epochs, 
            callbacks         = [model_checkpoint_callback],
            verbose           = 1,
            shuffle           = True,
        )

        return history

    def evaluate(self, directories:Directories, filenames:Filenames,
                jet_charge_kappa:float, preprocessing_details:str,
                image_dataloader:DataLoader,
                labels:np.ndarray):

        predicted_probability_is_down_quark = np.empty(len(labels))
        for (i, batch) in enumerate(image_dataloader):
            batch_as_array = np.array(batch)
            predicted_probability_is_down_quark[i*len(batch):(i+1)*len(batch)] = self.model.predict(batch_as_array)[:,1]
        
        eval_dir  = directories.save_dir_for_kappa(jet_charge_kappa)

        filename = filenames.model_result_filename_template(jet_charge_kappa, preprocessing_details)
        
        just_down_quark_labels = labels[:,1]
        down_quark_efficiency_roc_and_sic(filenames.data_details.energy_gev, filenames.data_details.data_year,
                                    jet_charge_kappa,
                                    predicted_probability_is_down_quark, just_down_quark_labels, 
                                    eval_dir, filename)

    def save(self, directories:Directories, filenames:Filenames, 
             jet_charge_kappa:float, preprocessing_details:str):
        
        filename       = filenames.saved_model_filename(jet_charge_kappa, preprocessing_details)
        save_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        save_filepath  = os.path.join(save_directory, filename)

        self.model.save(save_filepath)


