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
        return "_(year_{})_(energy_{}_gev)_".format(self.data_year, self.energy_gev)

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

    def dataset_details_str(self):
        return dataset_details_str(self.dataset_details)

    def training_directory(self):
        return os.path.join(self.save_data_directory, self.dataset_details_str(), "training")
    def training_image_directory(self):
        return os.path.join(self.training_directory(), "images")
    def training_label_directory(self):
        return os.path.join(self.training_directory(), "labels")

    def testing_directory(self):
        return os.path.join(self.raw_data_directory, self.dataset_details_str(),  "testing")
    def testing_image_directory(self):
        return os.path.join(self.testing_directory(), "images")
    def testing_label_directory(self):
        return os.path.join(self.testing_directory(), "labels")

class Filenames(object):
    def __init__(self, data_details:DataDetails):
        self.data_details = data_details

    def data_details_str(self):
        return dataset_details_str(self.data_details)

    def saved_model_filename(self, kappa:float):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        without_file_format  = "CNN"
        without_file_format += str(self.data_details)
        without_file_format += "(kappa_{})_".format(kappa)
        without_file_format += "(saved_{})_".format(timestamp)

        return without_file_format + ".keras"

    def roc_curve_filename(self, kappa:float):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        without_file_format  ="CNN_ROC"
        without_file_format += str(self.data_details)
        without_file_format += "(kappa_{})_".format(kappa)
        without_file_format += "(saved_{})_".format(timestamp)

        return without_file_format + ".png"
        