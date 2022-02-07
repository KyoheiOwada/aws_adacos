from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
import my_dataset
import glob


class ArcMarginProduct(nn.Module):
  def __init__(self, in_features, out_features, s=30.0, m=0.60, easy_margin=False):
    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight) # ???あｔoで調べて

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m #???

  def forward(self, input, label):
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0,1))
    phi = cosine * self.cos_m - sine * self.sin_m #加法定理でcos(th + m)を求める
    if self.easy_margin:
      phi = torch.where(cosine > 0, phi, cosine)
    else:
      phi = torch.where(cosine > self.th, phi, cosine - self.mm)

    one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot.scatter_(1,label.view(-1,1).long(), 1)
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= self.s
    print(output)

    return output


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
    x = self.Adacos_layer(x, y)

    return x

if __name__ == '__main__':

  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

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

  #for i, fname in enumerate(glob.glob("./dataset_csv/*.csv")):
  input_file_path = './train.csv'
  val_file_path = './val.csv'
  imgDataset = my_dataset.MyDataset(input_file_path,
                                    transform=transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  val_data = my_dataset.MyDataset(val_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  #print(imgDataset.__getitem__(0)[0].size())
  
  # load imagesi
  print("loading images...")
  #train_data, test_data = train_test_split(imgDataset, test_size=0)
  train_loader = torch.utils.data.DataLoader(imgDataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
  #val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False,num_workers=0, pin_memory=True)

  #GPU = True
  #device = torch.device("cuda" if GPU else "cpu")

  #net = CNN()
  print("loading_the model")
  #print(torch.load('model_weight.pth'))
  net.load_state_dict(torch.load('./model_weight/adacos_weight1.pth'))
  
  print(net)
  #net = net.to(device)
  #criterion = nn.CrossEntropyLoss()
  #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  #net.train()
  for epoch in range(1, 500+1):
    torch.save(net.state_dict(), "./model_weight/adacos_weight2.pth")
    print('saved the model.')
    net.train()
    for batch_idx, (image, label) in enumerate(train_loader):
      print("iteration:", batch_idx)
      optimizer.zero_grad()
      image, label = Variable(image).cuda(0), Variable(label).cuda(0)
      optimizer.zero_grad()
      output = net(image, label)
      #print("output_size is \t", output.size())
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      print('epoch: {}\t Loss: {}'.format(epoch , loss.data))
    """
    test_loss = 0
    correct = 0
    net.eval()
    for (image, label) in val_loader:
      image, label = Variable(image.float(), volatile=True).cuda(0), Variable(label).cuda(0)
      output = net(image, label)
      #print(output.size())
      test_loss += criterion(output, label).data
      pred = output.data.max(1, keepdim=True)[1]
      print('true label: {}\t pred label: {}'.format(label, pred))
      correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(val_loader.dataset)
    print('\ntestset: average loss:{:.4f}, accuracy:{}/{} ({:.0f}%)\n'.format(test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

    f = open('accuracy_log.txt','a')
    f.write(str(100. * correct / len(val_loader.dataset)) + '\n')
    f.close()

    if (100. * correct / len(val_loader.dataset)) > 90:
      break
    """
  print("training has been done.\n")
  torch.save(net.state_dict(), "./model_weight/adacos_weight2.pth")
