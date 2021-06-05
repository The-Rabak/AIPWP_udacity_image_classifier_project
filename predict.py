import functions.functions as myF

input_args = myF.get_predict_default_input_args()

model, optimizer, criterion, dataloaders,  hyper_params = myF.load_rabak_network_checkpoint(input_args.checkpoint)

category_names = myF.get_category_names_from_file(input_args.category_names)


predictions, classes = myF.predict(input_args.image, model, topk=input_args.top_k, use_gpu=input_args.gpu)

labels = [category_names[classes[i]] for i in range(len(classes))]
actual_label = category_names[input_args.image.split('/')[len(input_args.image.split('/')) - 2:-1].pop()]
#print(f'actual label is: {actual_label}, predictions are: \n', ''.join([f'label: {label}, probability: {ps} \n' for label, ps in zip(labels, predictions)]))

results_data_frame = myF.get_results_dataframe(predictions, labels)
print(f'\n actual label is: {actual_label}, predictions are: \n')
print(results_data_frame.to_string(index=False))

