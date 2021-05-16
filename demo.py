import torch
import torchvision


def run(model_path, image_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()


if __name__ == "__main__":
    run('./model/v3-small-row-11.pth', '')