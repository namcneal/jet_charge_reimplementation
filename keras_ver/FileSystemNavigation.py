import os


class Directories(object):
    def __init__(self, root_dir:str, raw_data_dir:str, image_dir:str, save_dir:str):
        # References the repository root directory
        self.cnn_project_root_directory = root_dir

        self.subdirectories_with_imports : list[str] = []

        self.subdirectories_with_imports.append(self.cnn_project_root_directory)
        for subdir_name in ["jet_data_tools", 
                            "jet_data_tools/jet_images", 
                            "keras_ver/cnn",]:
            self.subdirectories_with_imports.append(os.path.join(self.cnn_project_root_directory, subdir_name))

        self.raw_data_directory  = data_dir
        self.image_directory     = image_dir
        self.save_data_directory = save_dir

    def training_directory(self):
        return os.path.join(self.raw_data_directory, "training")
    def training_image_directory(self):
        return os.path.join(self.training_directory(), "images")
    def training_label_directory(self):
        return os.path.join(self.training_directory(), "labels")

    def testing_directory(self):
        return os.path.join(self.raw_data_directory, "testing")
    def testing_image_directory(self):
        return os.path.join(self.testing_directory(), "images")
    def testing_label_directory(self):
        return os.path.join(self.testing_directory(), "labels")

class Filenames(object):
    def __init__(self, data_year:int, energy_gev:int, kappa:float):
        self.data_year  = data_year
        self.energy_gev = energy_gev
        self.kappa      = kappa