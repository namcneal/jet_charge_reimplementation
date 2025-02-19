##

import os

def dataset_details_str(data_year:int, energy_gev:int, kappa:float):
    return "(data_year_{})_(energy_{}_GeV)_(jet_charge_kappa_{})".format(data_year, energy_gev, kappa)

class Directories(object):
    def __init__(self, repo_root_dir:str, raw_data_dir:str, image_dir:str, save_dir:str,
                 data_year:int, energy_gev:int, kappa:float):
        self.data_year  = data_year
        self.energy_gev = energy_gev
        self.kappa      = kappa
        self.dataset_details = dataset_details_str(data_year, energy_gev, kappa)

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

    def training_directory(self):
        return os.path.join(self.save_data_directory, self.dataset_details, "training")
    def training_image_directory(self):
        return os.path.join(self.training_directory(), "images")
    def training_label_directory(self):
        return os.path.join(self.training_directory(), "labels")

    def testing_directory(self):
        return os.path.join(self.raw_data_directory, self.dataset_details,  "testing")
    def testing_image_directory(self):
        return os.path.join(self.testing_directory(), "images")
    def testing_label_directory(self):
        return os.path.join(self.testing_directory(), "labels")

class Filenames(object):
    def __init__(self, data_year:int, energy_gev:int, kappa:float):
        self.data_year  = data_year
        self.energy_gev = energy_gev
        self.kappa      = kappa
        self.dataset_details = dataset_details_str(data_year, energy_gev, kappa)

    def saved_model_filename(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        without_file_format = "CNN_" * self.dataset_details *  "_(saved_{})".format(timestamp)

        return without_file_format + ".keras"

    def roc_curve_filename(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        without_file_format = "CNN_ROC_" * self.dataset_details *  "_(saved_{})".format(timestamp)

        return without_file_format + ".png"
        