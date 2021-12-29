import csv
import numpy as np
import os
import os.path
from typing import Callable
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class BaseSgVisionDataset(VisionDataset):
    """
    BaseSgVisionDataset
    """

    def __init__(self, root: str, sample_loader: Callable = default_loader, target_loader: Callable = None,
                 collate_fn: Callable = None, valid_sample_extensions: tuple = IMG_EXTENSIONS,
                 sample_transform: Callable = None, target_transform: Callable = None):
        """
        Ctor
            :param root:
            :param sample_loader:
            :param target_loader:
            :param collate_fn:
            :param valid_sample_extensions:
            :param sample_transform:
            :param target_transform:
        """
        super().__init__(root=root, transform=sample_transform, target_transform=target_transform)
        self.samples_targets_tuples_list = list(tuple())
        self.classes = []
        self.valid_sample_extensions = valid_sample_extensions
        self.sample_loader = sample_loader
        self.target_loader = target_loader
        self._generate_samples_and_targets()

        # IF collate_fn IS PROVIDED IN CTOR WE ASSUME THERE IS A BASE-CLASS INHERITANCE W/O collate_fn IMPLEMENTATION
        if collate_fn is not None:
            self.collate_fn = collate_fn

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        raise NotImplementedError

    def __len__(self):
        """

        :return:
        """
        return len(self.samples_targets_tuples_list)

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets - An abstract method that fills the samples and targets members of the class
        """
        raise NotImplementedError

    def _validate_file(self, filename: str) -> bool:
        """
        validate_file
            :param filename:
            :return:
        """
        for valid_extension in self.valid_sample_extensions:
            if filename.lower().endswith(valid_extension):
                return True

        return False

    @staticmethod
    def numpy_loader_func(path):
        """
        _numpy_loader_func - Uses numpy load func
            :param path:
            :return:
        """
        return np.load(path)

    @staticmethod
    def text_file_loader_func(text_file_path: str, inline_splitter: str = ' ') -> list:
        """
        text_file_loader_func - Uses a line by line based code to get vectorized data from a text-based file
            :param text_file_path:  Input text file
            :param inline_splitter: The char to use in order to separate between different VALUES of the SAME vector
                                    please notice that DIFFERENT VECTORS SHOULD BE IN SEPARATE LINES ('\n') SEPARATED
            :return: a list of tuples, where each tuple is a vector of target values
        """
        if not os.path.isfile(text_file_path):
            raise ValueError(" Error in text file path")

        with open(text_file_path, "r", encoding="utf-8") as text_file:
            targets_list = [tuple(map(float, line.split(inline_splitter))) for line in text_file]

        return targets_list


class DirectoryDataSet(BaseSgVisionDataset):
    """
    DirectoryDataSet - A PyTorch Vision Data Set extension that receives a root Dir and two separate sub directories:
                        - Sub-Directory for Samples
                        - Sub-Directory for Targets

    """

    def __init__(self, root: str,
                 samples_sub_directory: str, targets_sub_directory: str, target_extension: str,
                 sample_loader: Callable = default_loader, target_loader: Callable = None, collate_fn: Callable = None,
                 sample_extensions: tuple = IMG_EXTENSIONS, sample_transform: Callable = None,
                 target_transform: Callable = None):
        """
        CTOR
            :param root:                    root directory that contains all of the Data Set
            :param samples_sub_directory:   name of the samples sub-directory
            :param targets_sub_directory:   name of the targets sub-directory
            :param sample_extensions:       file extensions for samples
            :param target_extension:        file extension of the targets
            :param sample_loader:           Func to load samples
            :param target_loader:           Func to load targets
            :param collate_fn:              collate_fn func to process batches for the Data Loader
            :param sample_transform:        Func to pre-process samples for data loading
            :param target_transform:        Func to pre-process targets for data loading
        """

        # INITIALIZING THE TARGETS LOADER TO USE THE TEXT FILE LOADER FUNC
        if target_loader is None:
            target_loader = self.text_file_loader_func

        self.target_extension = target_extension
        self.samples_dir_suffix = samples_sub_directory
        self.targets_dir_suffix = targets_sub_directory

        super().__init__(root=root, sample_loader=sample_loader, target_loader=target_loader,
                         collate_fn=collate_fn, valid_sample_extensions=sample_extensions,
                         sample_transform=sample_transform, target_transform=target_transform)

    def __getitem__(self, item):
        """
        getter method for iteration
            :param item:
            :return:
        """
        sample_path, target_path = self.samples_targets_tuples_list[item]
        sample = self.sample_loader(sample_path)
        target = self.target_loader(target_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets - Uses class built in members to generate the list of (SAMPLE, TARGET/S)
                                        that is saved in self.samples_targets_tuples_list
        """
        missing_sample_files, missing_target_files = 0, 0

        # VALIDATE DATA PATH
        samples_dir_path = self.root + os.path.sep + self.samples_dir_suffix
        targets_dir_path = self.root + os.path.sep + self.targets_dir_suffix

        if not os.path.exists(samples_dir_path) or not os.path.exists(targets_dir_path):
            raise ValueError(" Error in data path")

        # ITERATE OVER SAMPLES AND MAKE SURE THERE ARE MATCHING LABELS
        for sample_file_name in os.listdir(samples_dir_path):
            sample_file_path = samples_dir_path + os.path.sep + sample_file_name
            if os.path.isfile(sample_file_path) and self._validate_file(sample_file_path):
                sample_file_prefix = str(sample_file_name.split('.')[:-1][0])

                # TRY TO GET THE MATCHING LABEL
                matching_target_file_name = sample_file_prefix + self.target_extension
                target_file_path = targets_dir_path + os.path.sep + matching_target_file_name
                if os.path.isfile(target_file_path):
                    self.samples_targets_tuples_list.append((sample_file_path, target_file_path))

                else:
                    missing_target_files += 1
            else:
                missing_sample_files += 1

        for counter_name, missing_files_counter in [('samples', missing_sample_files),
                                                    ('targets', missing_target_files)]:
            if missing_files_counter > 0:
                print(__name__ + ' There are ' + str(missing_files_counter) + ' missing  ' + counter_name)


class ListDataset(BaseSgVisionDataset):
    """
    ListDataset - A PyTorch Vision Data Set extension that receives a file with FULL PATH to each of the samples.
                  Then, the assumption is that for every sample, there is a * matching target * in the same
                  path but with a different extension, i.e:
                        for the samples paths:  (That appear in the list file)
                                                    /root/dataset/class_x/sample1.png
                                                    /root/dataset/class_y/sample123.png

                        the matching labels paths:  (That DO NOT appear in the list file)
                                                    /root/dataset/class_x/sample1.ext
                                                    /root/dataset/class_y/sample123.ext
    """

    def __init__(self, root, file, sample_loader: Callable = default_loader, target_loader: Callable = None,
                 collate_fn: Callable = None, sample_extensions: tuple = IMG_EXTENSIONS,
                 sample_transform: Callable = None, target_transform: Callable = None, target_extension='.npy'):
        """
           CTOR
               :param root:                    root directory that contains all of the Data Set
               :param file:                    Path to the file with the samples list
               :param sample_extensions:       file extension for samples
               :param target_extension:        file extension of the targets
               :param sample_loader:           Func to load samples
               :param target_loader:           Func to load targets
               :param collate_fn:              collate_fn func to process batches for the Data Loader
               :param sample_transform:        Func to pre-process samples for data loading
               :param target_transform:        Func to pre-process targets for data loading
           """

        if target_loader is None:
            target_loader = self.numpy_loader_func

        self.list_file_path = file
        self.loader = sample_loader
        self.target_loader = target_loader
        self.extensions = sample_extensions
        self.target_extension = target_extension

        super().__init__(root, sample_loader=sample_loader, target_loader=target_loader,
                         collate_fn=collate_fn, sample_transform=sample_transform,
                         valid_sample_extensions=sample_extensions,
                         target_transform=target_transform)

    def __getitem__(self, item):
        """
        Args:
            item (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample_path, target_path = self.samples_targets_tuples_list[item]
        sample = self.loader(sample_path)
        target = self.target_loader(target_path)[0]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets
        """
        file = open(self.root + os.path.sep + self.list_file_path, "r", encoding="utf-8")

        reader = csv.reader(file)
        data = [row[0] for row in reader]

        for f in data:
            path = self.root + os.path.sep + f
            target_path = path[:-4] + self.target_extension
            if self._validate_file(path) and os.path.exists(target_path):
                self.samples_targets_tuples_list.append((path, target_path))
