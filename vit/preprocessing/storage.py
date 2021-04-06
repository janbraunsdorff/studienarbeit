import pickle


def saveImage(file_name, data, path):
  filehandler = open(path + file_name, 'wb')
  pickle.dump(data, filehandler)


def loadImage(file_name, path):
  filehandler = open(path + file_name, 'rb') 
  data = pickle.load(filehandler)
  return data