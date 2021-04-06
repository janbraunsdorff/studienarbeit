import vit.model.config as conf
import torchvision.transforms as transforms
import torchvision
import torch
from vit.model.vit import ViT
from vit.trainer import Trainer

transform = transforms.Compose(
    [
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.CIFAR100(root='./vit/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_szie, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./vit/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_szie, shuffle=False)

model = ViT()
model = model.to(conf.device)

trainer = Trainer(model, trainloader, testloader)
trainer.fit()