import cv2
import torchvision.transforms as transforms
import torch
import pandas as pd
from os import listdir
from os.path import isfile, join
from v3.storage import loadImage, saveImage
from torch.utils.data import DataLoader
import sys
import numpy as np


def process_store_image_train(path_to_training_annotation, path_to_training_data, path_to_store):
    df_trainig = pd.read_csv(path_to_training_annotation)

    train = []
    counter = 5
    print('start processing')
    sys.stdout.flush()
    for index, row in df_trainig.iterrows():
        train.append(create_date_point(id = row[0], age=row[1], sex=float(row[2] if 1 else 0), path=path_to_training_data))
        if index % 500 == 0 and index != 0:
            print("saved processing {:5} / {:5} | Daten Punkte zu speichern: {}".format(index, len(df_trainig), len(train)))
            sys.stdout.flush()
            saveImage('/train01-' + str(counter) + '00.obj', train, path_to_store)
            train = []
            counter += 5

    print("saved processing {:5} / {:5} | Daten Punkte zu speichern: {}".format(index, len(df_trainig), len(train)))
    sys.stdout.flush()
    saveImage('/train01-' + str(counter) + '00.obj', train, path_to_store)


def process_store_image_val(path_to_validation_annotation, path_to_validatoin_data, path_to_store):
    df_trainig = pd.read_csv(path_to_validation_annotation)
    test = []
    counter = 5

    print('start processing')
    sys.stdout.flush()
    for index, row in df_trainig.iterrows():
        test.append(create_date_point(id=row[0], age=row[2], sex=float(row[1] if 1 else 0), path=path_to_validatoin_data))
        if index % 500 == 0 and index != 0:
            print("saved processing {:5} / {:5} | Daten Punkte zu speichern: {}".format(index, len(df_trainig), len(test)))
            sys.stdout.flush()
            saveImage('/validation01-' + str(counter) + '00.obj', test, path_to_store)
            test = []
            counter += 5

    # Save last Images, which are less than full collection of 500
    print("saved processing {:5} / {:5} | Daten Punkte zu speichern: {}".format(index, len(df_trainig), len(test)))
    sys.stdout.flush()
    saveImage('/validation01-' + str(counter) + '00.obj', test, path_to_store)

def load_data(path, batch_size):
    train_data = []
    val_data = []

    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in files:
        #print('Load ' + file)
        if not (file.startswith('train01-') or file.startswith('validation01-')):
            continue
        data = loadImage('/' + file, path)
        if file.startswith('train01-'):
            train_data.extend(data)
        elif file.startswith('validation01-'):
            val_data.extend(data)

    print('train_data size: ' + str(len(train_data)), end=" ")
    print('val_data size: ' + str(len(val_data)), end=" ")

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size)

    return train_loader, val_loader


def processImages(img_path, resize_to=750, reduce_to=500, out_to=299):
    img = cv2.imread(img_path)
    size_target = resize_to
    img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    x = y = (size_target - reduce_to) // 2
    h = w = reduce_to
    img = bgr[y:y + h, x:x + w]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = cv2.resize(img, (out_to, out_to), interpolation=cv2.INTER_LINEAR)

    
    return img


def img_to_tensor(base_path):
    img = processImages(img_path=base_path)
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t_img = torch.tensor(arr)
    t_img_normalize = t_img.unsqueeze(0)

    return t_img_normalize


def create_date_point(id, age, path, sex):
    point = (img_to_tensor(path + str(id) + ".png"), float(age), sex)
    return point