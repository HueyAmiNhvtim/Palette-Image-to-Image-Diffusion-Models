import torch.utils.data as data
from torchvision.transforms import v2
from PIL import Image
import os
import torch
import numpy as np
import csv

from pathlib import Path
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_path):
    """Generate a list of relative paths of the images
    Args:
        data_path (str): path to the file containing paths of images or path to the folder containing images
    Returns:
        list[np.ndarray]: A list of relative paths to images 
    """
    # Maybe make the flist here if possible.
    print(f"[INFO] data_path: {data_path}") 
    if os.path.isfile(data_path):   
        images = [i for i in np.genfromtxt(data_path, dtype=str, encoding='utf-8')]
    else:
        images = []   
        assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def make_dataset_from_custom_csv(csv_path: str):
    # Maybe we should just grab the images's path...., and then proceed to extract the cropped image
    # as actual data suitable for feeding into Palette  
    """Generate a list of relative paths of the whole images
    Args:
        csv_path (str): the path to the annotating csv_file 
    """
    
    pass


def pil_loader(path):
    return Image.open(path).convert('RGB')


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = v2.Compose([
                v2.Resize((image_size[0], image_size[1])),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass            
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class InpaintStampDataset(data.Dataset):
    # INCOMPLETE.
    # WE HAVE TO MAKE SURE TO TURN ALL PARTS WITH ALPHA 0 to 255 for black.
    # No, that can be taken care of by data generation part. Yeah, that part is done.
    def __init__(self, data_root: str, img_type="png", mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        """Initalizing Dataset for giving out cropped images of the document to send into the model
        Args:
            data_root (str): The path to the imgs folder of stamps
            img_type (str): Type of image. Defaults to png
            mask_config (dict, optional): _description_. Defaults to {}.
            data_len (int, optional): How much data you wanna use. Defaults to -1.
            image_size (list, optional): Size of input to the model. Defaults to [256, 256].
            loader (_type_, optional): For turning images into PIL Image objects. Defaults to pil_loader.
        """
        self.data_root = data_root
        self.img_type = img_type
        self.csv_data = []
        # Get the parent folder path
        parent_path, _ = os.path.split(data_root)
        annotating_csv_path = os.path.join(parent_path, "data_annotating_file_coco.csv")
        
        with open(annotating_csv_path, mode="r") as f:
            csv_reader = csv.DictReader(f=f)
            for line in csv_reader:
                self.csv_data.append(line)
        
        
        # We're gonna use coco data_annotating_file because that way, we don't have to convert other bbox_coordinate 
        # system back into COCO format for PIL's crop
        
        if data_len > 0:
            self.csv_data = self.csv_data[:int(data_len)]
        else:
            self.csv_data = self.csv_data
        self.tfs = v2.Compose([
                v2.Resize((image_size[0], image_size[1])),
                v2.PILToTensor(),  
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        

    def __getitem__(self, index):
        csv_data = self.csv_data[index] 
        # Get the corresponding origin of the csv data here!
        doc_id = csv_data["id"]
        target_name = csv_data["original"]
        
        doc_path = os.path.join(self.data_root, f"{doc_id}.{self.img_type}")
        target_path = os.path.join("", f"{target_name}")  # change "" to config.TEMPLATE_PATH when moved to the other project
        # A bit cursed because the target_name already has image type so yeah....
        
        img_coords = (csv_data["xmin_crop"], csv_data["ymin_crop"], csv_data["xmax_crop"], csv_data["ymax_crop"])

        ret = {}
        # Crop da thing here from the generated document
        target_doc = self.loader(target_path)
        target_data = target_doc.crop(box=img_coords)
        
        og_doc = self.loader(doc_path)
        img_data = og_doc.crop(box=img_coords)  # We also have to grab stuff from the original template inspiration too!
        
        img = self.tfs(img_data)  # Actual cropped image, now time to get da mask
        target = self.tfs(target_data)  # Note, this only works when the transformations don't contain random augmentations...
        
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = target
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = target_name # I think. Because the line as seen in other Datasets just extract the filename
        return ret

    def __len__(self):
        return len(self.csv_data)

    def get_mask(self, bbox_coord: tuple[int, int, int, int]=None):
        """Get the mask for a given image size

        Args:
            bbox_coord (tuple[int, int, int, int], optional):  Coordinates of the bounding box to mask off.
                                                               The bbox_coord is in xyxy format.
                                                               Defaults to None.
        Raises:
            NotImplementedError: if mask_mode is not accounted for in the below images
        Returns:
            Mask tensor of the image
        """
        if bbox_coord is None:
            if self.mask_mode == 'bbox':
                mask = bbox2mask(self.image_size, random_bbox())
            elif self.mask_mode == 'center':
                h, w = self.image_size
                mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
            elif self.mask_mode == 'irregular':
                mask = get_irregular_mask(self.image_size)
            elif self.mask_mode == 'free_form':
                mask = brush_stroke_mask(self.image_size)
            elif self.mask_mode == 'hybrid':
                regular_mask = bbox2mask(self.image_size, random_bbox())
                irregular_mask = brush_stroke_mask(self.image_size, )
                mask = regular_mask | irregular_mask
            elif self.mask_mode == 'file':
                pass
            else:
                raise NotImplementedError(
                    f'Mask mode {self.mask_mode} has not been implemented.')
        else:
            if self.mask_mode == 'hybrid_custom_crop':
                # [INCOMPLETE] Use bbox2mask with your bounding box coordinate here! and do same thing as hybrid_mode
                pass
            elif self.mask_mode == 'custom_crop':
                # [INCOMPLETE] Use bbox2mask with your bounding box coordinate here!            
                pass
            else:
                raise NotImplementedError(
                    f'Mask mode {self.mask_mode} has not been implemented.')
                
        return torch.from_numpy(mask).permute(2,0,1)
    

class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


