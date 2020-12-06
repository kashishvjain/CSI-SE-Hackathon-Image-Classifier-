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


def index(request):
    return render(request, 'home.html')

def backend(request):
    global error_prediction

    if request.method == "POST":
        print('HII')
        print(request.POST['tag'])
        ind,val = request.POST['tag'].split(' ')
        print("PRINTING prediction")
        error_prediction.append([ind,val])
        print(error_prediction)
        messages.success(request,"Request successfully submitted. Sent for Retraining!")
        return redirect("/backend")

    print("Here in backend")
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
    print(output)
    probablities=torch.exp(output)/(torch.sum(torch.exp(output),1)).reshape(-1,1)
    probablities=probablities.detach().numpy()
    print(probablities)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())

    for pred in preds:
        prediction.append(classes[pred])
        print(classes[pred])
    img_directory=os.listdir(test_dir+'images')
    img_with_labels={}
    # for img,label in zip(img_directory,prediction):
    #     img_with_labels[img[:-4]]=label
    for img_direc,prob in zip(img_directory,probablities):
        print("Hey man")
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
    print(path_prob)
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
        print(form)
        if form.is_valid():
            object = form.save(commit=False)
            object.save()

            messages.success(request, "Your entry has been noted")
            return redirect("/backend")
        else:
            print("(**********")
            print("Hi")
            print("*******")
            return redirect("/")


    else:
        form = img()
        print("uuuuuuuuu")
        return render(request, 'video.html', {"form": form})
