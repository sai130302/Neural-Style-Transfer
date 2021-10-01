import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
def get_style_model_and_losses(style_img, content_img,cnn = vgg19,content_layers = [7],style_layers = [0,2,5,7,10]) :
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    i = 0
    for layer_num,layer in enumerate(cnn):
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)
        
        model.add_module("layer"+str(i),layer)
        i += 1
        if layer_num in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = utils.ContentLoss(target)
            model.add_module("content_loss" + str(i),content_loss)
            i += 1
            content_losses.append(content_loss)

        if layer_num in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = utils.StyleLoss(target_feature)
            model.add_module("style_loss "+str(i),style_loss)
            i += 1
            style_losses.append(style_loss) 

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], utils.ContentLoss) or isinstance(model[i], utils.StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

def run_style_transfer(content_img, style_img, input_img,num_steps=300,
                       style_weight=1000000, content_weight=1):

    model, style_losses, content_losses = get_style_model_and_losses(style_img, content_img)
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
