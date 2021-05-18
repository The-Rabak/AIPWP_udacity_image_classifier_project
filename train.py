from torch import nn
from collections import OrderedDict

from consts.consts import optimizer_name, test_image_size as image_size
import functions.functions as myF

input_args = myF.get_default_input_args()
model_name = input_args.arch
hyper_params = OrderedDict([('epochs', int(input_args.epochs)), ('lr', float(input_args.learning_rate)), ('classifier_hidden_layers', list(map(int, str(input_args.hidden_units).strip('[]').split(','))))])
model_input_sizes = myF.get_model_input_sizes(model_name)
hyper_params['model_input_sizes'] = model_input_sizes


data_dir = input_args.images_dir
train_transforms = myF.get_training_transofrmers(image_size)

test_transforms = myF.get_validation_transformers(image_size)
train_dir = myF.get_train_dir_path(data_dir)
valid_dir = myF.get_validation_dir_path(data_dir)
test_dir = myF.get_test_dir_path(data_dir)

train_data = myF.get_datasets_imagefolder(train_dir, train_transforms)
validation_data = myF.get_datasets_imagefolder(valid_dir, test_transforms)
test_data = myF.get_datasets_imagefolder(test_dir, test_transforms)

image_datasets = OrderedDict([('train', train_data), ('test', test_data), ('validate', validation_data)])

trainloader = myF.get_data_loader(image_datasets['train'])
validationloader = myF.get_data_loader(image_datasets['validate'])
testloader = myF.get_data_loader(image_datasets['test'])

dataloaders = OrderedDict([('train', trainloader), ('test', testloader), ('validate', validationloader)])
print(hyper_params)
classifier = myF.get_rabak_classifier(hyper_params)
model = myF.get_frozen_model(model_name, classifier)

criterion = nn.NLLLoss()
optimizer = myF.get_optimizer(model, model_name, hyper_params['lr'])
train_losses, test_losses = myF.train(model, dataloaders['train'], dataloaders['validate'], criterion, optimizer, hyper_params['epochs'], use_gpu=input_args.gpu)

def save_trained_model(checkpoint_dir, checkpoint_name):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.hyper_params = hyper_params
    model.input_sizes = model_input_sizes
    myF.save_model(model, model_name, optimizer, optimizer_name, criterion, dataloaders, checkpoint_dir + "/" + checkpoint_name)

save_trained_model(input_args.save_dir, input_args.checkpoint_name)

print("model trained and saved successfully")

