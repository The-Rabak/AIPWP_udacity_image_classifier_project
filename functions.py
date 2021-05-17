import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
from PIL import Image
from rabak_net import RabakNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_training_transofrmers(image_size=(224,224)):
    return transforms.Compose([transforms.RandomRotation(13),
                                       transforms.RandomResizedCrop(image_size[0]),
                                       transforms.RandomGrayscale(p=0.2), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
     
def get_validation_transformers(image_size=(224,224)):
    return transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(image_size[0]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def get_datasets_imagefolder(image_dir, transformers):
    return datasets.ImageFolder(image_dir, transform=transformers)

def get_data_loader(dataset, batch_size=40, shuffle = True, pin_memory = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

def train(model, trainloader, testloader, criterion, optimizer, epochs = 5, print_every_n = 40 ):

    steps = 0
    running_loss = 0
    model = model.to(device)
    running_losses, test_losses = [], []
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            images, labels = images.to(device) , labels.to(device)

            output = model.forward(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()            
            running_loss += loss.item()

            if steps % print_every_n == 0:
                #turn off gradients and speed up the model
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validate(model, testloader, criterion)
                print(f'Epoch {e+1} / {epochs}',
                      f'Training Loss: {(running_loss / print_every_n):.3f}',
                      f'Test Loss: {(test_loss / len(testloader)):.3f}',
                      f'Test Accuracy: {(accuracy / len(testloader)):.3f}'
                )

                running_losses.append(running_loss / print_every_n)
                test_losses.append(test_loss / len(testloader))
                running_loss = 0
                #turn gradients back on
                model.train()
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


def save_model(model, model_name, optimizer, optimizer_name, loss, dataloaders,  file_name):
    checkpoint = {'hyper_params': model.hyper_params,
              'class_to_idx': model.class_to_idx,
              'optimizer_dict': optimizer.state_dict(),
              'optimizer': optimizer_name,
              'loss': loss,
              'dataloaders': dataloaders,
              'rabak_classifier': model.fc,    
              'model': model_name,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, file_name)
    
def get_random_image_and_label(path):
    
    subdirs=os.listdir(path)
    rand_dir = np.random.choice(subdirs)
    
    pics = os.listdir(path + "/" + rand_dir)
    pic = np.random.choice(pics)
    
    return pic, rand_dir

def process_image(image, add_batch=True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        im = im.resize((256,256))
        width, height = im.width, im.height
        
        new_width, new_height = 224, 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        im = im.crop((round(left), round(top), round(right), round(bottom)))
        
        np_image = np.array(im)
        
        #divide by the max value to get an array of values between 0 and 1
        np_normalized = np_image / 255
        
        #subtract the means from each color channel, then divide by the standard deviation
        np_normalized = (np_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        #reorder the dimensions so that the color channel is first, as pytorch expects
        np_normalized = np_normalized.transpose(2,0,1)
        
        #add a fourth dimension to indicate batch size for torchvision
        if add_batch:
            np_normalized = np_normalized[np.newaxis,:]
    
        # Turn into a torch tensor
        image = torch.from_numpy(np_normalized)
        image = image.float()
        return image

                                
def process_transform_images(img_path):
                                
    with Image.open(img_path) as img:
        transformer = get_validation_transformers()
        #return as float tensor with extra dimension to indicate batch size, which the model expects
        return transformer(img).unsqueeze(0).type(torch.FloatTensor)

def predict(image_path, model, topk=5):
    image = process_transform_images(image_path)
    image = image.to("cpu")
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        
    ps = torch.exp(output)
    outputs = ps.topk(5)
    
    top_ps, top_classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    top_classes = [idx_to_class[top_classes[i]] for i in range(top_classes.size)]
    return top_ps, top_classes


def display_prediction(classes, class_names, predictions, test_image, test_image_dir):
    
    labels = [class_names[val] for val in classes]

    fig, ((ax1,ax2)) = plt.subplots(figsize=(6,7), nrows=2)
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


def validate_model(model, dataset, test_device = None):
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
    print(f'Accuracy of the network: {(accuracy / len(dataset) * 100):.3f}%')

            

def load_rabak_network_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['model']](pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], strict = False)

    for param in model.parameters():
        param.requires_grad = False
    model.fc = checkpoint['rabak_classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.__dict__[checkpoint['optimizer']](model.fc.parameters(), lr=checkpoint['hyper_params']['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])           
    loss = checkpoint['loss']
    dataloaders = checkpoint['dataloaders']
    hyper_params = checkpoint['hyper_params']
    del checkpoint
    return model, optimizer, loss, dataloaders, hyper_params

