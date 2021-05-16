import torch
import torchvision


from v3.processing import img_to_tensor


def run(model_path, image_path, sex):
    print('load model ...', end='')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print('*done*')
    print('eval model ...', end='')
    model.eval()
    print('*done*')

    print('convert image to tensor ...', end='')
    img  = img_to_tensor(image_path).unsqueeze(0)
    print('*done*')
    gender = torch.Tensor([sex])

    print('prediction: ', end='')
    print(round(model(img, gender).item(), 3), 'Monate')




if __name__ == "__main__":
    run('./model/v3-small-row-11.pth', '../data/boneage-test-dataset/4360.png', 0)
    print('done')