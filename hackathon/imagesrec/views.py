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
images_id=[]
#from .model import *
import torch
from django.conf import settings
from django.contrib.staticfiles.storage import staticfiles_storage
url = staticfiles_storage.path('vgg_model2')
vgg16=torch.load(url)
#print(model1)
def index(request):
    return render(request, 'home.html')

def backend(request):
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

# prepare data loaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()
    # get sample outputs
    output = vgg16(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea','street']
    for pred in preds:
        print(classes[pred])
    
    return HttpResponse("Hello there")
    
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