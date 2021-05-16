import torch
import torchvision

from v3.processing import img_to_tensor


def run(model_path, image_path, sex):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    img  = img_to_tensor(image_path).squeeze(0)
    gender = torch.Tensor([sex]).squeeze(0)

    print(img.shape, gender.shape)
    print(model(img, gender))




if __name__ == "__main__":
    run('./model/v3-small-row-11.pth', '../data/boneage-test-dataset/4360.png', 0)
    print('done')