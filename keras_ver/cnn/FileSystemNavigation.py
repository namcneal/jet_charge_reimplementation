from datetime import datetime 
import os
import sys

higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

##
from JetImages import PreprocessingSpecification

def two_channel_preprocessing_str(channel_one:PreprocessingSpecification, channel_two:PreprocessingSpecification):
    return "channel_one_{}_channel_two_{}".format(channel_one, channel_two)

class DataDetails(object):
    def __init__(self, data_year:int, energy_gev:int):
        self.data_year  = data_year
        self.energy_gev = energy_gev

    def __str__(self):
        return "(year_{})_(energy_{}_gev)_".format(self.data_year, self.energy_gev)

class Directories(object):
    def __init__(self, repo_root_dir:str, 
                raw_data_dir:str, image_dir:str, save_dir:str,
                data_details:DataDetails):

        self.dataset_details = data_details

        # References the repository root directory
        self.repository_root_directory = repo_root_dir

        self.subdirectories_with_imports : list[str] = []

        self.subdirectories_with_imports.append(self.repository_root_directory)
        for subdir_name in ["jet_data_tools", 
                            "jet_data_tools/jet_images", 
                            "keras_ver/cnn",]:
            self.subdirectories_with_imports.append(os.path.join(self.repository_root_directory, subdir_name))

        self.raw_data_directory  = raw_data_dir
        self.image_directory     = image_dir
        self.save_data_directory = save_dir

    def save_dir_for_kappa(self, kappa:float):
        return os.path.join(self.save_data_directory, "kappa_{}".format(kappa))

    def output_data_super_directory(self, dataset_type:str,
                                    kappa:float, 
                                    channel_one:PreprocessingSpecification, 
                                    channel_two:PreprocessingSpecification):
                                    
        if dataset_type not in ["training", "validation", "testing"]:
            raise ValueError("dataset_type must be one of 'training', 'validation', 'testing'")

        return os.path.join(self.image_directory, str(self.dataset_details), dataset_type)

    def output_image_directory(self, dataset_type:str, 
                                kappa:float,
                                channel_one:PreprocessingSpecification,
                                channel_two:PreprocessingSpecification):
        super_dir = self.output_data_super_directory(dataset_type, kappa, channel_one, channel_two)
        return os.path.join(super_dir, "images")

    def output_label_directory(self, dataset_type:str,
                                kappa:float,
                                channel_one:PreprocessingSpecification,
                                channel_two:PreprocessingSpecification):

        super_dir = self.output_data_super_directory(dataset_type, kappa, channel_one, channel_two)
        return os.path.join(super_dir, "labels")

    def all_output_data_directories(self, kappa:float, 
                                    channel_one:PreprocessingSpecification, 
                                    channel_two:PreprocessingSpecification):

        for dataset_type in ["training", "validation", "testing"]:
            yield (self.output_image_directory(dataset_type, kappa, channel_one, channel_two),
                   self.output_label_directory(dataset_type, kappa, channel_one, channel_two))

class Filenames(object):
    def __init__(self, data_details:DataDetails):
        self.data_details = data_details

    def data_details_str(self):
        return dataset_details_str(self.data_details)



    def model_result_filename_template(self, kappa:float, 
                                channel_one:PreprocessingSpecification, 
                                channel_two:PreprocessingSpecification):

        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        spec_str  = self.two_channel_preprocessing_str(channel_one, channel_two)

        without_file_format  = "CNN_"
        without_file_format += str(self.data_details)
        without_file_format += "(kappa_{})_".format(kappa)
        without_file_format += "({})_".format(spec_str)
        without_file_format += "(saved_{})_".format(timestamp)

        return without_file_format

    def saved_model_filename(self, kappa:float,
                             channel_one:PreprocessingSpecification, 
                             channel_two:PreprocessingSpecification):

        return self.model_result_filename_template(kappa, channel_one, channel_two) + ".keras"

    def roc_curve_filename(self, kappa:float,
                           channel_one:PreprocessingSpecification, 
                           channel_two:PreprocessingSpecification):

        return self.model_result_filename_template(kappa, channel_one, channel_two) + ".png"