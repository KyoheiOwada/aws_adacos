import torch
class MyDataset(torch.utils.data.Dataset):

  def __init__(self, image_list, transform=None):
    self.image_list = image_list
    self.transform = transform

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    out_data = self.image_list[idx]
    if self.transform:
      out_data = self.transform(out_data)
    out_label = 0
    return out_data, out_label
