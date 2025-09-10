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
            sys.path.append(self.subdirectories_with_imports[-1])

        self.raw_data_directory  = raw_data_dir
        self.image_directory     = image_dir
        self.save_data_directory = save_dir

    def save_dir_for_kappa(self, kappa:float):
        return os.path.join(self.save_data_directory, "kappa_{}".format(kappa))

    def output_data_super_directory(self, dataset_type:str, kappa:float, preprocessing_details:str):
        if dataset_type not in ["training", "validation", "testing"]:
            raise ValueError("dataset_type must be one of 'training', 'validation', 'testing'")
        return os.path.join(self.image_directory, str(self.dataset_details), "kappa_{}".format(kappa), "{}_".format(preprocessing_details), dataset_type)

    def output_image_directory(self, dataset_type:str, kappa:float,preprocessing_details:str):
        super_dir = self.output_data_super_directory(dataset_type, kappa, preprocessing_details)
        return os.path.join(super_dir, "images")

    def output_label_directory(self, dataset_type:str, kappa:float, preprocessing_details:str):
        super_dir = self.output_data_super_directory(dataset_type, kappa, preprocessing_details)
        return os.path.join(super_dir, "labels")

    def all_output_data_directories(self, kappa:float, preprocessing_details:str):
        for dataset_type in ["training", "validation", "testing"]:
            yield (self.output_image_directory(dataset_type, kappa, preprocessing_details),
                   self.output_label_directory(dataset_type, kappa, preprocessing_details))

class Filenames(object):
    def __init__(self, data_details:DataDetails):
        self.data_details = data_details

    def data_details_str(self):
        return dataset_details_str(self.data_details)



    def model_result_filename_template(self, kappa:float, preprocessing_details:str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        without_file_format  = "CNN_"
        without_file_format += str(self.data_details)
        without_file_format += "(kappa_{})_".format(kappa)
        without_file_format += "({})_".format(preprocessing_details)
        without_file_format += "(saved_{})_".format(timestamp)

        return without_file_format


    def saved_model_filename(self, kappa:float, preprocessing_details:str):
        return self.model_result_filename_template(kappa, preprocessing_details) + ".keras"
    
    def saved_eval_plot_data(self, kappa:float, preprocessing_details:str):
        return self.model_result_filename_template(kappa, preprocessing_details) + "efficiencies.npz"

