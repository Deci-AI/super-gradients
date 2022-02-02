import collections


class DaliClassificationDataLoader(collections.Iterator):
    """
    DataLoader wrapper for dali.
    """
    def __init__(self, dali_loader):
        self.dali_loader = dali_loader

    def __next__(self):
        batch = self.dali_loader.__next__()
        images = batch[0]['data']
        labels = batch[0]['label'][:, 0]
        return images, labels.long()

    def __len__(self):
        return len(self.dali_loader)