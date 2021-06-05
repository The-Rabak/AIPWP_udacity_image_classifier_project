model_name = "resnet101"
optimizer_name = "AdamW"
data_dir = 'assets/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
test_image_size = (224,224)
checkpoint_dir = "model_checkpoints"
checkpoint_file_name = "rabakClassifier2.pth"
checkpoint_path = checkpoint_dir + "/" + checkpoint_file_name
default_lr = 0.001
default_hidden_units = 512
default_epochs = 4
default_is_gpu_on = True
default_prediction_topk = 3

default_cat_to_names_file = "assets/cat_to_name.json"