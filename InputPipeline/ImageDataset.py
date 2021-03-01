import os.path
from InputPipeline.base_dataset import BaseDataset, get_params, get_transform, get_downscale_transform
from InputPipeline.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(BaseDataset):
    def initialize(self, opt, fixed=False):
        self.fixed = fixed
        self.opt = opt
        self.root = opt.dataroot

        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.unl_folder + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.B_paths)

    def __getitem__(self, index):
        ### input B (real images)
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        params = get_params(self.opt, B.size)
        transform_B = get_transform(self.opt, params, fixed=self.fixed)
        B_tensor = transform_B(B)
        B_temp = B_tensor

        input_dict = {'image': B_tensor, 'path': B_path}
        return input_dict

    def __len__(self):
        return len(self.B_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'ImageDataset'
