higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

##

import os

from data_loading import dataset_details_str, DataDetails

class Directories(object):
    def __init__(self, repo_root_dir:str, raw_data_dir:str, image_dir:str, save_dir:str,
                 data_details:DataDetails):
        self.dataset_details

        # References the repository root directory
        self.repository_root_directory = repo_root_dir

        self.subdirectories_with_imports : list[str] = []

        self.subdirectories_with_imports.append(self.repository_root_directory)
        for subdir_name in ["jet_data_tools", 
                            "jet_data_tools/jet_images", 
                            "keras_ver/cnn",]:
            self.subdirectories_with_imports.append(os.path.join(self.repository_root_directory, subdir_name))

        self.raw_data_directory  = data_dir
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

        without_file_format  = "CNN_"
        without_file_format += self.data_details_str()
        without_file_format += "_kappa_{}".format(kappa)
        without_file_format += "_saved_{}".format(timestamp)

        return without_file_format + ".keras"

    def roc_curve_filename(self, kappa:float):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        without_file_format  ="CNN_ROC_"
        without_file_format += self.dataset_details_str()
        without_file_format += "_(saved_{})".format(timestamp)

        return without_file_format + ".png"
        