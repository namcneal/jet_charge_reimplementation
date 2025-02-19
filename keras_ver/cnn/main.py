import argparse
import os 
import sys 

# at keras_ver/cnn/FilesystemNavigation.py
project_root_directory = os.path.abspath(os.path.join("..", ".."))
sys.path.append(project_root_directory)

from FilesystemNavigation import Directories, Filenames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw-jet-data-dir', type=str,   required=True)
    parser.add_argument('--min_data_seed',  type=int,   required=True)
    parser.add_argument('--max_data_seed',  type=int,   required=True)
    parser.add_argument('--data-year', type=int,   required=True)
    parser.add_argument('--energ-gev', type=int,   required=True)
    parser.add_argument('--kappa',     type=float, required=True)
    parser.add_argument('--image-dir', type=str,   required=True)
    parser.add_argument('--save-dir',  type=str,   required=True)
    parser.add_argument('--batch_size', type=int,   required=True, default=512)
    parser.add_argument('--num_epochs', type=int,   required=True, default=35)
    parser.add_argument('--val_pct',    type=float, required=True, default=1/8)

    args = parser.parse_args()

    main(directories_navigation, Filenames())



def configure_system(args:argparse.Namespace):

    directories = Directories(
                    root_dir     = project_root_directory,
                    raw_data_dir = args.raw_jet_data_dir,
                    image_dir    = args.image_dir,
                    save_dir     = args.save_dir
    )
    
    filenames = Filenames(
                    data_year = args.data_year,
                    energy_gev = args.energy_gev,
                    kappa = args.kappa
    )

    for sub_dir in directories_navigation.subdirectories_with_imports:
        sys.path.append(sub_dir)

    return directories, filenames


def main(args:argparse.Namespace):

    directories, filenames = configure_system(args)

    from generate_images import JetChargeAttributes, generate_and_save_all_images
    from data_loading    import MemmapDataset
    from cnn import CNNSpecification, CNN

    jet_data_seeds = range(args.min_data_seed, args.max_data_seed + 1)
    num_jet_data_seeds = len(jet_data_seeds)

    generate_and_save_all_images(directories, filenames, jet_data_seeds)
    training_dataset = MemmapDataset.datasets_from_memmaps(directories.training_image_directory(), directories.training_label_directory())
    training_data_loader = DataLader(training_dataset)

    cnn_specification = CNNSpecification.default()
    cnn_model         = CNN(cnn_specification)

    training_history = cnn_model.train_model(training_data_loader, args.val_pct, args.batch_size, args.num_epochs)

    # evaluate_cnn()



