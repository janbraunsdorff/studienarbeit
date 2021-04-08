import torchvision.transforms as transforms
import vit.model.config as conf

data_augmentation = transforms.Compose(
    [
            transforms.RandomRotation(degrees=10),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=30)
            ], p=0.7),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.1, 0.7))
    ]
)