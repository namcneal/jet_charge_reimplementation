
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]
# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from FileSystemNavigation import Directories, Filenames

##

class MemmapDataset(Dataset):
    def __init__(self, image_memmap, label_memmap):
        self.images = image_memmap
        self.labels = label_memmap

        if len(self.images) != len(self.labels):
            raise ValueError("Number of images and labels must be the same.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])

    @staticmethod
    def datasets_from_memmaps(image_directory:str, label_directory:str):    
        images = mmap_ninja.np_open_existing(image_directory)
        labels = mmap_ninja.np_open_existing(label_directory)
        return MemmapDataset(images, labels)




        

