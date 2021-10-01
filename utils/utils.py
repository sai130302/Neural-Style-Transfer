import torch
import cv2 as cv
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

def get_image(img_path,target_shape,device = "cpu"):
    """
    Arguments:
    img_path -- file path of the image 
    target_shape -- tuple (nH,nW) to reshape the image
    device -- "cpu" or "conda"

    Returns: 
    image -- image transformed into tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean,std = mean.to(device),std.to(device)
    image = Image.open(img_path)
    loader = transforms.Compose([
        transforms.Resize(target_shape),
        transforms.ToTensor(),
        #transforms.Normalize(mean,std)
    ])
    image = loader(image).to(device).unsqueeze(0)
    return image

def tensor_to_image(tensor,show = False,title = "Image"):
    """
    Arguments:
    tensor -- tensor of image
    show -- boolean for plotting the image
    title -- for the plot

    Returns: 
    img -- tensor transformed into a images
    """
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = (transforms.ToPILImage())(img)
    if show :
        plt.figure()
        plt.imshow(img)
        plt.title(title) 
        plt.show()
    return img

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(img_fm):
    a, b, c, d = img_fm.size() 
    features = img_fm.view(b, c * d) 
    G = torch.mm(features, features.t()) 
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input