import argparse
import os 
import sys 

from torch.utils.data import DataLoader
repository_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      repository_root_directory
                    ]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

# at keras_ver/cnn/FilesystemNavigation.py
from FileSystemNavigation import Directories, Filenames, DataDetails

def configure_system(args:argparse.Namespace):
    data_details = DataDetails(args.data_year, args.energy_gev)

    directories = Directories(
                    repo_root_dir= repository_root_directory,
                    raw_data_dir = args.raw_jet_data_dir,
                    image_dir    = args.image_dir,
                    save_dir     = args.save_dir,
                    data_details = data_details
    )

    for sub_dir in directories.subdirectories_with_imports:
        sys.path.append(sub_dir)

    return directories
    
def run_one_kappa(args:argparse.Namespace, directories:Directories, jet_data_seeds:range, kappa:float,):
    from generate_images import JetChargeAttributes, generate_and_save_all_images
    from data_loading    import MemmapDataset
    from model import CNNSpecification, CNN
    
    filenames = Filenames(directories.dataset_details)

    generate_and_save_all_images(directories, filenames, jet_data_seeds, kappa)

    training_dataset   = MemmapDataset.datasets_from_memmaps(directories.training_image_directory(),   directories.training_label_directory())
    validation_dataset = MemmapDataset.datasets_from_memmaps(directories.validation_image_directory(), directories.validation_label_directory())

    training_data_loader   = DataLoader(training_dataset,   shuffle=True, batch_size=args.batch_size)
    validation_data_loader = DataLoader(validation_dataset, shuffle=True, batch_size=args.batch_size)

    cnn_specification = CNNSpecification.default()
    cnn_model         = CNN(cnn_specification)

    training_history = cnn_model.train(directories, filenames, 
                                        kappa,
                                        training_data_loader,
                                        validation_data_loader,
                                        args.batch_size, args.num_epochs)

    testing_dataset = MemmapDataset.datasets_from_memmaps(directories.testing_image_directory(), directories.testing_label_directory())
    
    # MUST be batched for the Keras usage of the DataLoader in the model.predict method
    # Otherwise, 'get_tensor_spec' in 'keras/src/trainers/data_adapters/torch_data_loader_adapter.py' will throw an error. 
    # It expects a list of batches, otherwise we get  "TypeError: 'TensorSpec' object is not iterable"
    testing_batch_size        = args.batch_size
    testing_images_dataloader = DataLoader(testing_dataset.just_images(), batch_size=testing_batch_size)
    testing_labels            = testing_dataset.labels

    cnn_model.evaluate(directories, filenames,
                       kappa,
                       testing_images_dataloader, 
                       testing_dataset.labels)

    cnn_model.save(directories, filenames, kappa)

def main(args:argparse.Namespace):
    directories = configure_system(args)

    all_jet_data_seeds = range(args.min_data_seed, args.max_data_seed + 1)
    all_kappas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for _, kappa in enumerate(all_kappas):
        run_one_kappa(args, directories, all_jet_data_seeds, kappa)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw-jet-data-dir', type=str, required=True)
    parser.add_argument('--min-data-seed',  type=int,   required=True)
    parser.add_argument('--max-data-seed',  type=int,   required=True)
    parser.add_argument('--data-year',      type=int,   required=True)
    parser.add_argument('--energy-gev',     type=int,   required=True)
    parser.add_argument('--image-dir',      type=str,   required=True)
    parser.add_argument('--regen-images',   type=bool,  default=False)
    parser.add_argument('--save-dir',       type=str,   required=True)
    parser.add_argument('--batch-size',     type=int,   default=512)
    parser.add_argument('--num-epochs',      type=int,  default=35)

    main(parser.parse_args())

