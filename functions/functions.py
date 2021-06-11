import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import os
from torchvision import datasets, transforms, models
from PIL import Image
import time
from collections import OrderedDict
import pandas as pd

from classes.rabak_net import RabakNetwork
from consts.consts import  data_dir, default_is_gpu_on, default_epochs, default_lr, default_hidden_units, \
    checkpoint_dir, checkpoint_file_name, checkpoint_path, default_prediction_topk, default_cat_to_names_file
from classes.CommandArgs import CommandArgs
from classes.ParseArgs import ParseArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_frozen_model(model_name, classifier):
    if model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "vgg13":
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if "resnet" in model_name:
        model.fc = classifier
    elif "vgg" in model_name:
        model.classifier = classifier

    return model


def get_model_input_sizes(model_name):
    if "resnet" in model_name:
        model_input_sizes = OrderedDict([('input', 2048), ('output', 102)])
    elif "vgg" in model_name:
        model_input_sizes = OrderedDict([('input', 25088), ('output', 102)])
    return model_input_sizes


def get_rabak_classifier(hyper_params):
    return RabakNetwork(hyper_params['model_input_sizes']['input'],
                        hyper_params['model_input_sizes']['output'],
                        hyper_params['classifier_hidden_layers'])


def get_optimizer(model, model_name, lr):
    if "resnet" in model_name:
        optimizer = optim.AdamW(model.fc.parameters(), lr)
    elif "vgg" in model_name:
        optimizer = optim.AdamW(model.classifier.parameters(), lr)
    return optimizer


def get_train_default_input_args():
    arch_models = {
        "vgg13": "vgg13",
        "vgg16": "vgg16",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
    }
    arch_arg = CommandArgs(["-a", "--arch"], str, choices=arch_models,
                           help="choose a proper architecture model from one of these: " + ", ".join(
                               arch_models.keys()),
                           default="resnet101"
                           )

    save_dir_arg = CommandArgs(["-d", "--save_dir"], str,
                               help="provide a directory for saving and loading models",
                               default=checkpoint_dir
                               )

    checkpoint_name_arg = CommandArgs(["-c", "--checkpoint_name"], str,
                                      help="provide a directory for saving and loading models",
                                      default=checkpoint_file_name
                                      )

    learn_rate_arg = CommandArgs(["-l", "--learning_rate"], float,
                                 help="provide a learning rate scalar for training models",
                                 default=default_lr
                                 )

    hidden_units_arg = CommandArgs(["-u", "--hidden_units"], str,
                                   help="provide a number of hidden units for hidden model",
                                   default=default_hidden_units
                                   )

    epochs_arg = CommandArgs(["-e", "--epochs"], int,
                             help="provide a number of epochs for training loop",
                             default=default_epochs
                             )

    gpu_arg = CommandArgs(["-g", "--gpu"], bool,
                          help="should we use gpu (if available) for training?",
                          default=default_is_gpu_on
                          )

    images_dir_arg = CommandArgs([], bool,
                                 help="please provide a directory with valid, test and train subdirs for model",
                                 default=data_dir
                                 )
    ArgsParser = ParseArgs([arch_arg, save_dir_arg, learn_rate_arg, hidden_units_arg, epochs_arg,
                            gpu_arg, checkpoint_name_arg, images_dir_arg])
    ArgsParser.set_args()
    return ArgsParser.get_args()


def get_predict_default_input_args():

    default_validation_dir = get_validation_dir_path(data_dir)

    test_image, label_dir = get_random_image_and_label(default_validation_dir)
    default_image_path = default_validation_dir + "/" + label_dir + "/" + test_image
    arch_arg = CommandArgs([], str,
                           help="provide a valid image path to predict on",
                           default=default_image_path
                           )

    save_dir_arg = CommandArgs(["--checkpoint"], str,
                               help="provide a full path for and loading model",
                               default=checkpoint_path
                               )

    checkpoint_name_arg = CommandArgs(["--top_k"], str,
                                      help="how many classes do you want returned in prediction",
                                      default=default_prediction_topk
                                      )

    learn_rate_arg = CommandArgs(["--category_names"], str,
                                 help="provide a mapping of categories to names",
                                 default=default_cat_to_names_file
                                 )

    gpu_arg = CommandArgs(["-g", "--gpu"], bool,
                          help="should we use gpu (if available) for inference?",
                          default=default_is_gpu_on
                          )

    ArgsParser = ParseArgs([arch_arg, save_dir_arg, learn_rate_arg,
                            gpu_arg, checkpoint_name_arg])
    ArgsParser.set_args()
    return ArgsParser.get_args()


def get_training_transofrmers(image_size=(224, 224)):
    return transforms.Compose([transforms.RandomRotation(13),
                               transforms.RandomResizedCrop(image_size[0]),
                               transforms.RandomGrayscale(p=0.2),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])


def get_validation_transformers(image_size=(224, 224)):
    return transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(image_size[0]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])


def get_datasets_imagefolder(image_dir, transformers):
    return datasets.ImageFolder(image_dir, transform=transformers)


def get_data_loader(dataset, batch_size=40, shuffle=True, pin_memory=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def get_train_dir_path(main_dir):
    return main_dir + "/train"


def get_test_dir_path(main_dir):
    return main_dir + "/test"


def get_validation_dir_path(main_dir):
    return main_dir + "/valid"


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every_n=40, use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if device == 'cuda':
        torch.cuda.empty_cache()

    train_start_time = time.perf_counter()
    steps = 0
    running_loss = 0
    model = model.to(device)
    running_losses, test_losses = [], []
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every_n == 0:
                # turn off gradients and speed up the model
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validate(model, testloader, criterion)
                print(f'Epoch {e + 1} / {epochs}',
                      f'Training Loss: {(running_loss / print_every_n):.3f}',
                      f'Test Loss: {(test_loss / len(testloader)):.3f}',
                      f'Test Accuracy: {(accuracy / len(testloader)):.3f}'
                      )

                running_losses.append(running_loss / print_every_n)
                test_losses.append(test_loss / len(testloader))
                running_loss = 0
                # turn gradients back on
                model.train()

    print(f'classifier runtime: {time.perf_counter() - train_start_time} seconds')
    return running_losses, test_losses


def validate(model, validationloader, criterion):
    accuracy = 0
    test_loss = 0

    for images, labels in validationloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += torch.mean(equality.type(torch.FloatTensor))

    return test_loss, accuracy


def save_model(model, model_name, optimizer, optimizer_name, loss, dataloaders, file_name):
    checkpoint = {'hyper_params': model.hyper_params, 'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(), 'optimizer': optimizer_name, 'loss': loss,
                  'dataloaders': dataloaders, 'model': model_name, 'state_dict': model.state_dict(),
                  'rabak_classifier': model.fc if "resnet" in model_name else model.classifier}

    torch.save(checkpoint, file_name)


def get_random_image_and_label(path):
    subdirs = os.listdir(path)
    rand_dir = np.random.choice(subdirs)

    pics = os.listdir(path + "/" + rand_dir)
    pic = np.random.choice(pics)

    return pic, rand_dir


def process_image(image, add_batch=True):

    with Image.open(image) as im:
        im = im.resize((256, 256))
        width, height = im.width, im.height

        new_width, new_height = 224, 224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        im = im.crop((round(left), round(top), round(right), round(bottom)))

        np_image = np.array(im)

        # divide by the max value to get an array of values between 0 and 1
        np_normalized = np_image / 255

        # subtract the means from each color channel, then divide by the standard deviation
        np_normalized = (np_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # reorder the dimensions so that the color channel is first, as pytorch expects
        np_normalized = np_normalized.transpose(2, 0, 1)

        # add a fourth dimension to indicate batch size for torchvision
        if add_batch:
            np_normalized = np_normalized[np.newaxis, :]

        # Turn into a torch tensor
        image = torch.from_numpy(np_normalized)
        image = image.float()
        return image


''' 
A more condensed and elegant form of process_image
uses existing validation transformers instead of performing calculations from scratch
'''
def process_transform_images(img_path):
    with Image.open(img_path) as img:
        transformer = get_validation_transformers()
        # return as float tensor with extra dimension to indicate batch size, which the model expects
        return transformer(img).unsqueeze(0).type(torch.FloatTensor)

def get_results_dataframe(predictions, labels):
    return pd.DataFrame({
        'labels': pd.Series(data=labels),
        'predictions': pd.Series(data=predictions, dtype='float64').round(3)
    })

def predict(image_path, model, topk=5, use_gpu=True):
    image = process_transform_images(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)

    ps = torch.exp(output)
    outputs = ps.topk(int(topk))

    top_ps, top_classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    top_classes = [idx_to_class[top_classes[i]] for i in range(top_classes.size)]
    return top_ps, top_classes


def display_prediction(classes, class_names, predictions, test_image, test_image_dir):
    labels = [class_names[val] for val in classes]

    fig, ((ax1, ax2)) = plt.subplots(figsize=(6, 7), nrows=2)
    with Image.open(test_image) as img:
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(class_names[test_image_dir])

    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, predictions)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()

    plt.show()


def validate_model(model, dataset, test_device=None):
    test_device = test_device if test_device else device
    accuracy = 0
    model = model.to(test_device)

    with torch.no_grad():
        for images, labels in dataset:
            images, labels = images.to(test_device), labels.to(test_device)

            output = model.forward(images)
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += torch.mean(equality.type(torch.FloatTensor))
    print(f'Model accuracy at: {(accuracy / len(dataset) * 100):.3f}%')

def get_category_names_from_file(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_rabak_network_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['model']
    model = models.__dict__[model_name](pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    if "resnet" in model_name:
        model.fc = checkpoint['rabak_classifier']
        model_params = model.fc.parameters()
    else:
        model.classifier = checkpoint['rabak_classifier']
        model_params = model.classifier.parameters()

    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.__dict__[checkpoint['optimizer']](model_params, lr=checkpoint['hyper_params']['lr'])
    # for some reason, loading the state_dict was setting the optimizer on cpu while the rest of the tensors were on cuda,
    # this was causing an error so I had to drop it.
    # optimizer.load_state_dict(checkpoint['optimizer_dict'])
    loss = checkpoint['loss']
    dataloaders = checkpoint['dataloaders']
    hyper_params = checkpoint['hyper_params']
    # recommended for avoiding cuda GPU memory overload errors
    del checkpoint
    return model, optimizer, loss, dataloaders, hyper_params
