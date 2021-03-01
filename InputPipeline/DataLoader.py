import torch.utils.data

def CreateDataLoader(opt, batchSize = None, shuffle=None, fixed = False, dataset = None):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    if batchSize == None:
        batchSize = opt.batchSize
    if shuffle == None:
        shuffle = not opt.serial_batches
    data_loader.initialize(opt, batchSize, shuffle, fixed, dataset)
    return data_loader

class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, batch_size, shuffle, fixed, dataset):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, fixed, dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def CreateDataset(opt, fixed, dataset = None):
    if dataset == "image":
        from InputPipeline.ImageDataset import ImageDataset as Dataset
    elif dataset == "mask":
        from InputPipeline.MaskDataset import MaskDataset as Dataset
    else:
        from InputPipeline.AlignedDataset import AlignedDataset as Dataset
    dataset = Dataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, fixed)
    return dataset
