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
from keras.layers import Dense, Embedding, GRU, Lambda

from JetImages import PreprocessingSpecification
from FileSystemNavigation import Directories, Filenames, DataDetails
from roc_sic_tools import down_quark_efficiency_roc

class RNN(object):
    def __init__(self, vector_length:int = -1,):
        self.vector_length = vector_length
        self.gru_nodes_per_layer = list(range(100,0,-10)) + [5]
        self.num_dense_nodes = 64  
        self.loss = 'binary_crossentropy'  
        self.opt  = 'adam'

    @staticmethod
    def create_model(self, comp=True, summary=True):
        model = Sequential()
        
        for nodes in self.gru_nodes_per_layer:
            model.add(GRU(nodes, return_sequences=True))

        # Output of GRU is a 3D tensor (batch_size, timesteps, features)
        # Averagingo over the sequence length (timesteps) to get a 2D tensor (batch_size, features)
        model.add(Lambda(lambda x: keras.ops.mean(x, axis=1)))  

        model.add(Dense(self.num_dense_nodes, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        if comp:
            model.compile(loss = self.loss, optimizer = self.opt, metrics = ['accuracy'])
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
        
        plot_dir  = directories.save_dir_for_kappa(jet_charge_kappa)

        plot_name = filenames.roc_curve_filename(jet_charge_kappa, preprocessing_details)
        
        just_down_quark_labels = labels[:,1]
        down_quark_efficiency_roc(filenames.data_details.energy_gev, filenames.data_details.data_year,
                                    jet_charge_kappa,
                                    predicted_probability_is_down_quark, just_down_quark_labels, 
                                    plot_dir, plot_name)

    def save(self, directories:Directories, filenames:Filenames, 
             jet_charge_kappa:float, preprocessing_details:str):
        
        filename       = filenames.saved_model_filename(jet_charge_kappa, preprocessing_details)
        save_directory = directories.save_dir_for_kappa(jet_charge_kappa)
        save_filepath  = os.path.join(save_directory, filename)

        self.model.save(save_filepath)


