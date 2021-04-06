import torchvision.transforms as transforms
import vit.model.config as conf

data_augmentation = transforms.Compose(
    [
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=0.2),
        transforms.RandomResizedCrop(size=(conf.image_size, conf.image_size))
    ]
)