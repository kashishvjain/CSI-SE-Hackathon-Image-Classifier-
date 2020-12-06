from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import otherDetails
from django.contrib import messages
from django.contrib import messages
from django.views.generic.edit import FormView
from torchvision import datasets, models, transforms
import torch
from .forms import img
from .model import *
import torch
from django.contrib.staticfiles.storage import staticfiles_storage

url = staticfiles_storage.path('vgg_model2')
print(url)
# vgg16 = models.vgg16(pretrained=True)
# print(vgg16)
# state_dict = torch.load("checkpoint.pth")
# print("Done")
# vgg16.load_state_dict(state_dict)
# print("finally done")
model1=torch.load(url)
print("Done")
print(model1)
def index(request):
    print("In view function")
    print("Successfull")
    return render(request, 'home.html')

def backend(request):
    return HttpResponse("Hello there")


def bulk(request):
    if request.method == "POST":
        # form = img(request.POST, request.FILES)
        # if form.is_valid():
        #     object = form.save(commit=False)
        #     object.save()

        #     messages.success(request, "Your entry has been noted")
        #     return redirect("/backend")
        # else:
        #     messages.success(request, "Sorry wrong")
        #     return redirect("/")
        my_file = request.FILES.get("file")
        otherDetails.objects.create(image = my_file)
        return redirect("/backend")



    else:
        form = img()
        return render(request, 'bulk.html')
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