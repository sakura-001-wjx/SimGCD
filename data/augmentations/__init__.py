from torchvision import transforms

import torch
# 用于获取适用于图像数据的训练和测试变换操作。该函数针对 ImageNet 数据集的图像预处理
def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:

        raise NotImplementedError

    return (train_transform, test_transform)