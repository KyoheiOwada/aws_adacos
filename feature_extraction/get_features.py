from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import my_dataset
from torch.utils.data import Dataset
import numpy as np
from model import model
from torchvision import models
import math

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.resnet = models.wide_resnet101_2(pretrained=True)
    #self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    print(self.resnet.fc)
    self.resnet.fc = nn.Linear(2048, 4096)
    #self.ArcFace_layer = ArcMarginProduct(1024, 2360, easy_margin=True)
    self.Adacos_layer = AdaCos(4096, 16758)
    #print(self.resnet)

  def forward(self, x, y):
    #x = x.view(-1, 25*25*128)
    x = self.resnet(x)
    fv = x
    x = self.Adacos_layer(x, y)

    return fv, x

if __name__=='__main__':
  db_file_path = 'csvfiles/detection_db.csv'
  pred_file_path = 'csvfile/detection_val.csv'
  ROOT_DIR = ""
  mini_batch = 1
  out_feature = 4096
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  database = my_dataset.MyDataset(db_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  database_loader = torch.utils.data.DataLoader(database, batch_size=mini_batch, shuffle=False,num_workers=0, pin_memory=True)
  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  #net = net.to(device)
  #net = nn.DataParallel(net)
  #model = model.TrainedNet("wide_resnet101")
  #model.load_state_dict(
  #        torch.load('model_weight/wideresnet101_weight.pth'))

  # define network model

  net = CNN()
  net = net.to(device)
  net = nn.DataParallel(net)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  print("load model weight\n")
  net.load_state_dict(torch.load('./model_weight/adacos_weight2.pth'))
  print("load compleated\n")

  net.eval()

  #feature_vectors_list = np.zeros((mini_batch,out_feature))
  feature_vectors_list = []
  #save latent codes
  for (image, label) in database_loader:
    image, label = Variable(image.float(), volatile=True).cuda(0), Variable(label).cuda(0)
    feature_vectors, _ = net(image, label)
    print(feature_vectors.data.cpu().device)
    feature_vectors = feature_vectors.data.cpu().numpy()
    feature_vectors_list.append(feature_vectors)
    #feature_vectors_list = np.concatenate((feature_vectors_list, feature_vectors))
  #feature_vectors_list = np.delete(feature_vectors_list, np.s_[0:mini_batch], 0)
  #print(feature_vectors_list[2])
  print(len(feature_vectors_list))
  feature_vectors_list = np.array(feature_vectors_list)
  print(feature_vectors_list.shape)
  print(feature_vectors_list[0].shape)
  np.save('./feature_vectors.npy', feature_vectors_list)

