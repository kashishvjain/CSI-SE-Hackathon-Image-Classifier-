from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import otherDetails
from django.contrib import messages
from django.contrib import messages
from django.views.generic.edit import FormView
from torchvision import datasets, models, transforms
import torch
from .forms import img
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns


import time
images_id=[]
error_prediction = []
#from .model import *
import torch
from django.conf import settings
from django.contrib.staticfiles.storage import staticfiles_storage
url = staticfiles_storage.path('vgg_model2')
vgg16=torch.load(url)
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea','street']
#print(model1)
img_with_labels={}
import shutil
# def create_my_model(input_layer,output_layer,hidden_array):
#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv1 = nn.Conv2d(1, 6, 3)
#             self.conv2 = nn.Conv2d(6, 16, 3)
#             # an affine operation: y = Wx + b
#             self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#             self.fc2 = nn.Linear(120, 84)
#             self.fc3 = nn.Linear(84, 10)

#         def forward(self, x):
#             # Max pooling over a (2, 2) window
#             x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#             # If the size is a square you can only specify a single number
#             x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#             x = x.view(-1, self.num_flat_features(x))
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x
def delete():
    folders=['media/imagesrec/images','media/imagesrec/train/forest','media/imagesrec/train/buildings','media/imagesrec/train/glacier',
    'media/imagesrec/train/sea','media/imagesrec/train/street','media/imagesrec/train/mountain']
    for folder in folders :
        folder = folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                     shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    global img_with_labels
    img_with_labels={}

def index(request):
    delete()
    return render(request, 'home.html')

def training(request):
    if request.method == "POST":
        lr=float(request.POST['lr'])
        batch_size = int(request.POST['bs'])
        epochs=int(request.POST['ep'])
        for image,label in img_with_labels.items():
            shutil.copy('media/imagesrec/images/'+image+'.jpg', 'media/imagesrec/train/'+label)
        train_dir ='media/imagesrec/train'
        #print(len(os.listdir(tp_dir)))
        # test_dir='imagesrec/images'
        data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor()])
        train_data = datasets.ImageFolder(train_dir, transform=data_transform)
        import torch.optim as optim
        from torch import nn
        num_workers=0
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=False)
        # specify loss function (categorical cross-entropy)
        criterion = nn.CrossEntropyLoss()
        # specify optimizer (stochastic gradient descent) and learning rate = 0.001
        optimizer = optim.SGD(vgg16.classifier.parameters(), lr=lr)
        print_every=1
        training_loss_list=[]
        ## TODO complete epoch and training batch loops
        def train(vgg16,train_loader,epochs=epochs,print_every=1):
            steps=0
            running_loss=0
            for e in range(epochs):
                vgg16.train()
                for image,labels in train_loader:
                    steps+=1
                    optimizer.zero_grad()
                    output=vgg16(image)
                    loss=criterion(output,labels)
                    loss.backward()
                    optimizer.step()
                    running_loss+=loss.item()
                    if steps % print_every == 0:
                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                            "Training Loss: {:.3f}.. ".format(running_loss/print_every))
                        training_loss_list.append((running_loss/print_every))
                        running_loss = 0
            return training_loss_list
        loss_list=train(vgg16,train_loader,epochs,print_every)
        plt.plot(loss_list);
        plt.xlabel('epochs')
        plt.ylabel('Losses')
        path_prob= 'media/imagesrec/'+'images/'+'graph.jpg'
       #print(path_prob)
        plt.savefig(path_prob)
        plt.cla()
        plt.close()
## These loops should update the classifier-weights of this model
## And track (and print out) the training loss over time
    return render(request,"train.html")

def backend(request):
    global error_prediction
    global img_with_labels
    if request.method == "POST":
        #print('HII')
        #print(request.POST['tag'])
        ind,val = request.POST['tag'].split(' ')
        #print("PRINTING prediction")
        img_with_labels[ind]=val
        # error_prediction.append([ind,val])
        # print(error_prediction)
        return redirect("/backend")


    if len(img_with_labels) != 0:
        return render(request,"backend.html",{'predict':img_with_labels})

    #print("Here in backend")
    test_dir ='media/imagesrec/'
    #print(len(os.listdir(tp_dir)))
    # test_dir='imagesrec/images'
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    batch_size = len(os.listdir(test_dir+'images'))
#earlier was 20
    num_workers=0
    prediction = []
# prepare data loaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=False)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()
    # get sample outputs
    output = vgg16(images)
    #print(output)
    probablities=torch.exp(output)/(torch.sum(torch.exp(output),1)).reshape(-1,1)
    probablities=probablities.detach().numpy()
    #print(probablities)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())

    for pred in preds:
        prediction.append(classes[pred])
        print(classes[pred])
    img_directory=os.listdir(test_dir+'images')

    # for img,label in zip(img_directory,prediction):
    #     img_with_labels[img[:-4]]=label
    for img_direc,prob in zip(img_directory,probablities):
        #print("Hey man")
        if (not os.path.exists(test_dir+'images/' + img_direc[:-4] + '_p.jpg')) and (img_direc[-5]!='p'):
            plotter(img_direc,prob,test_dir)


    for img,label in zip(img_directory,prediction):
        if len(img)<5 or img[-5]!='p':
            img_with_labels[img[:-4]]=label
   # return render(request,"backend.html",{'predict':prediction,'tags':os.listdir(test_dir+'images') })
    return render(request,"backend.html",{'predict':img_with_labels})


def plotter(img_direc,prob,test_dir):
    plt.bar(classes,prob);
    plt.xlabel('Classes')
    plt.ylabel('Relativistic Probablities')
    path_prob= test_dir+'images/'+img_direc[:-4] + '_p.jpg'
    #print(path_prob)
    plt.savefig(path_prob)
    plt.cla()
    plt.close()

def bulk(request):
    if request.method == "POST":
        my_file = request.FILES.get("file")
        otherDetails.objects.create(image = my_file)
        return redirect("/bulk")
    else:
        form = img()
        return render(request, 'bulk.html')
def redirection_backend(request):
    if request.method=="POST":
        return redirect("/backend")
def video(request):
    if request.method == "POST":
        form = img(request.POST, request.FILES)
        #print(form)
        if form.is_valid():
            object = form.save(commit=False)
            object.save()

            messages.success(request, "Your entry has been noted")
            return redirect("/backend")
        else:
            #print("(**********")
            #print("Hi")
            #print("*******")
            return redirect("/")


    else:
        form = img()
        #print("uuuuuuuuu")
        return render(request, 'video.html', {"form": form})
