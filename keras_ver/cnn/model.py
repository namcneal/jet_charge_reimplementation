from datetime import datetime
import os, sys
import pickle

higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

##
import numpy as np 
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from JetImages import PreprocessingSpecification
from FileSystemNavigation import Directories, DataDetails
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
                    learning_rate,
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
        self.learning_rate = learning_rate
        self.opt = opt

    @staticmethod
    def default():
        return CNNSpecification(
            image_size=33, 
            num_image_channels=2,

            num_conv_layers=3,
            conv_layer_num_filters    =[64,   64,   64],
            conv_layer_filter_sizes   =[8,    4,    4],
            conv_pooling_kernel_sizes =[2,    2,    2],
            conv_dropout_percents     =[0.18, 0.35, 0.35],

            intermediate_dense_size=64,
            intermediate_dense_dropout=0.35,
            intermediate_dense_activation='relu',

            num_final_dense_logits=2,
            final_dense_logits_activation='softmax',

            stopping_patience=5,
            loss='categorical_crossentropy',
            learning_rate=0.001,
            opt=keras.optimizers.Adam,
        )

class CNN(object):
    def __init__(self, specification:CNNSpecification):
        self.specification = specification
        self.model = CNN.create_model(self.specification)        

    @staticmethod
    def create_model(specification:CNNSpecification, comp=True, summary=True):
        model = Sequential()
        model.add(keras.Input(shape=(specification.num_image_channels, specification.image_size, specification.image_size)))

        params_each_layer = zip(specification.conv_layer_num_filters, 
                                specification.conv_layer_filter_sizes, 
                                specification.conv_pooling_kernel_sizes,
                                specification.conv_dropout_percents)

        for i,(num_filters, filter_size, pooling_kernel_size, dropout_percent) in enumerate(params_each_layer):
            kwargs : dict = {}

            # "channels_first" corresponds to inputs with shape (batch_size, channels, height, width)
            model.add(Conv2D(num_filters, filter_size, 
                            kernel_initializer = 'he_uniform', 
                            padding = 'valid',
                            activation = specification.conv_activation,
                            data_format = 'channels_first',
                            use_bias = True,
                            **kwargs)) 
            
            model.add(MaxPooling2D(pool_size = pooling_kernel_size, data_format = 'channels_first'))
            
            model.add(Dropout(dropout_percent))

        model.add(Flatten())
        model.add(Dense(specification.intermediate_dense_size, activation = specification.intermediate_dense_activation))
        model.add(Dropout(specification.intermediate_dense_dropout))

        model.add(Dense(specification.num_final_dense_logits, activation = specification.final_dense_logits_activation))

        if comp:

            model.compile(loss      = specification.loss, 
                          optimizer = specification.opt(specification.learning_rate), 
                          metrics = ['accuracy']
           )
                        
            if summary:
                model.summary()

        return model
    
    @classmethod
    def result_filename_template(cls, kappa:float, data_details:DataDetails, preprocessing_details:str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        without_file_format  = "CNN_"
        without_file_format += "(saved_{})_".format(timestamp)
        without_file_format += str(data_details)
        without_file_format += "(kappa_{})_".format(kappa)
        without_file_format += "({})_".format(preprocessing_details)

        return without_file_format

    @classmethod
    def saved_model_filename(cls, kappa:float, data_details:DataDetails, preprocessing_details:str):
        return cls.result_filename_template(kappa, data_details, preprocessing_details) + ".keras"

    @classmethod
    def efficiencies_filename(cls, kappa:float, data_details:DataDetails, preprocessing_details:str):
        return cls.result_filename_template(kappa, data_details, preprocessing_details) + "efficiencies.npz"

    @classmethod
    def training_history_filename(cls, kappa:float, data_details:DataDetails, preprocessing_details:str):
        return cls.result_filename_template(kappa, data_details, preprocessing_details) + "training_history_dict.pkl"

    def train(self, directories:Directories, data_details:DataDetails,
                jet_charge_kappa:float, preprocessing_details:str,
                training_image_label_data:DataLoader, 
                validation_image_label_data:DataLoader,
                batch_size:int, epochs:int):

        checkpoint_filename  = "checkpoint_" + self.saved_model_filename(jet_charge_kappa, data_details, preprocessing_details)
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

    def evaluate(self, directories:Directories, data_details:DataDetails,
                jet_charge_kappa:float, preprocessing_details:str,
                image_dataloader:DataLoader,
                labels:np.ndarray):

        all_images = None
        for (i, batch) in enumerate(image_dataloader):
            if i == 1:
                all_images = np.array(batch) # First batch
            else:
                all_images = np.concatenate((all_images, batch), axis=0)
        
        predicted_probability_is_down_quark    = np.empty(len(labels))
        predicted_probability_is_down_quark[:] = self.model.predict(all_images)[:,1]
        
        eval_dir  = directories.save_dir_for_kappa(jet_charge_kappa)
        filename  = self.efficiencies_filename(jet_charge_kappa, data_details, preprocessing_details)
        
        just_down_quark_labels = labels[:,1]
        down_quark_efficiency_roc_and_sic(data_details.data_year, data_details.energy_gev,
                                            jet_charge_kappa,
                                            predicted_probability_is_down_quark, just_down_quark_labels, 
                                            eval_dir, filename)
        
    def save(self, directories:Directories, data_details:DataDetails,
             jet_charge_kappa:float, preprocessing_details:str, training_history=None, plot_history=True):

        filename       = self.saved_model_filename(jet_charge_kappa, data_details, preprocessing_details)
        save_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        save_filepath  = os.path.join(save_directory, filename)
        self.model.save(save_filepath)

        if training_history is not None:
            history_filepath = os.path.join(save_directory, self.training_history_filename(jet_charge_kappa, data_details, preprocessing_details))

            with open(history_filepath, 'wb') as file_pi:
                pickle.dump(training_history.history, file_pi)

            if plot_history:
                # Plot the training and validation loss and accuracy 
                # in two different axes, top then bottom, in that order
                plt.figure(figsize=(8, 12))
                top_ax = plt.subplot(2, 1, 1)
                bottom_ax = plt.subplot(2, 1, 2)

                top_ax.plot(training_history.history['loss'], label='Training Loss', color='blue')
                top_ax.plot(training_history.history['val_loss'], label='Validation Loss', color='orange')

                bottom_ax.plot(training_history.history['accuracy'], label='Training Accuracy', color='blue')
                bottom_ax.plot(training_history.history['val_accuracy'], label='Validation Accuracy', color='orange')

                top_ax.set_title('Training and Validation Loss')
                top_ax.set_xlabel('Epochs')
                top_ax.set_ylabel('Loss')

                bottom_ax.set_title('Training and Validation Accuracy')
                bottom_ax.set_xlabel('Epochs')
                bottom_ax.set_ylabel('Accuracy')

                top_ax.legend()
                bottom_ax.legend()

                plt.tight_layout()
                plot_filepath = os.path.join(save_directory, self.result_filename_template(jet_charge_kappa, data_details, preprocessing_details) + "training_history.png")
                plt.savefig(plot_filepath)


    @classmethod
    def load_model(cls, directories:Directories, data_details:DataDetails,
                   jet_charge_kappa:float, preprocessing_details:str):

        filename       = cls.saved_model_filename(jet_charge_kappa, data_details, preprocessing_details)
        load_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        load_filepath  = os.path.join(load_directory, filename)

        loaded_model = keras.models.load_model(load_filepath)
        cnn = cls(CNNSpecification.default())
        cnn.model = loaded_model

        return cnn
    
    def load_training_history(cls, directories:Directories, data_details:DataDetails, jet_charge_kappa:float, preprocessing_details:str):

        history_filename = cls.training_history_filename(jet_charge_kappa, data_details, preprocessing_details)
        load_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        load_filepath  = os.path.join(load_directory, history_filename)

        with open(load_filepath, 'rb') as file_pi:
            history_dict = pickle.load(file_pi)

        return history_dict

        
    # """
    #     Tunable parameters:
    #     * num_conv_layers
    #     * conv_layer_num_filters 
    #     * conv_dropout_percents
    #     * intermediate_dense_size
    #     * intermediate_dense_dropout
    #     * learning_rate
    # """
    # class HyperparameterRanges(object):
    #     def __init__(self, specification:CNNSpecification):
    #         self.parameter_ranges = dict(
    #             num_conv_layers            = range(3, 7, 1),
    #             conv_layer_num_filters     = [32, 64, 96, 128],
    #             conv_dropout_percents      = range(0.1, 0.30, 0.02),
    #             intermediate_dense_size    = [32, 64, 128, 256],
    #             intermediate_dense_dropout = range(0.1, 0.5, 0.05),
    #             learning_rate              = range(1e-4, 101e-4,5e-4)
    #         )

    # def modify_specification(specification:CNNSpecification, options:HyperparameterRanges,  hp):
    #     setattr(specification, 'num_conv_layers', hp.Choice('num_conv_layers', options.parameter_ranges['num_conv_layers']))
    #     specification.conv_dropout_percents      = hp.Choice('conv_dropout_percents', options.parameter_ranges['conv_dropout_percents'])
    #     specification.intermediate_dense_size    = hp.Choice('intermediate_dense_size', [8, 16, 32, 64])
    #     specification.intermediate_dense_dropout = hp.Choice('intermediate_dense_dropout', [0.2, 0.35, 0.5])
    #     specification.learning_rate              = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    #     return specification
            
    # def lambda_create_from_hp(self, specification:CNNSpecification, options:HyperparameterRanges, comp=True, summary=True):
       
       

    #    lambda hp: CNN.create_model(specification, comp=comp, summary=summary)


