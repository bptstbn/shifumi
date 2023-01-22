from rock_paper_scissors_code_only import *
import torch


def show_history(hist_path):
    hist = load_history(hist_path)
    show_validation_loss(hist)
    show_training_accuracy(hist)


# first plot absolute accuracys in valitdation set next to each other
def compare_histories(path_hist_1, path_hist_2, compare_accuracy=True, compare_loss=True):
    history_no_dropouts = load_history(path_hist_1)
    history_wi_dropouts = load_history(path_hist_2)
    # this compares two histories
    if compare_accuracy:
        for key in ['total', 'rock', 'paper', 'scissors']:
            show_compare_training_accuracy([history_no_dropouts, history_wi_dropouts],
                                           ['without dropouts', 'with dropouts'], key=key, colors=['b-', 'g-'])
    if compare_loss:
        show_compare_training_loss([history_no_dropouts, history_wi_dropouts], ['without dropouts', 'with dropouts'],
                                   colors=['b-', 'g-'])


def get_all_model_iterations(model_name='pytk_rock_paper_scissors', dropouts=True, batch_norm=False,
                             start_iterations=10, stop_iterations=100, step=10):
    model_paths = []
    iter = start_iterations
    while (iter <= stop_iterations):
        path = f'./model_states/{model_name}_{iter}epoches__Dropouts_{str(dropouts)}__BatchNorm_{str(batch_norm)}.pt'
        model_paths.append(path)
        iter += step
    models = []
    for path in model_paths:
        model = RPS_CNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append({'model': model, 'name': path})
    return models


def test_accuracy_for_model_iterations(model_iterations, dataloader):
    hist = []
    for model in model_iterations:
        m = model['model']
        res = test_accuracy(m, dataloader)
        hist.append({'name': model['name'], 'accuracy_measures': res})
        print(model['name'] + str(res) + '\n\n')
    return hist


def load_data_from_pattern(file_pattern, batch_size=32, target_size=(64, 64), show_images=False):
    images, labels, data_paths = get_images_and_labels_from_filepattern(file_pattern, target_size)
    dataset = RPSDataset(images, labels, image_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if show_images:
        show_images_from_dataloader(loader)
    return loader


if __name__ == "__main__":
    ### Playing with the actual training history

    history_no_dropouts = './model_states/history__pytk_rock_paper_scissors_100epoches__Dropouts_False__BatchNorm_False'
    history_wi_dropouts = './model_states/history__pytk_rock_paper_scissors_100epoches__Dropouts_True__BatchNorm_False'

    #show_history(history_wi_dropouts)
    #show_history(history_no_dropouts)
    #compare_histories(history_no_dropouts, history_wi_dropouts)

    ### Playing with model versions to create histories

    # TODO build history for training and test
    # TODO download test and Preprocess
    # TODO train on no hands preprocessing
    #data_loader = load_data_from_pattern(file_pattern_training_data)

    #models_wi_dropouts = get_all_model_iterations(dropouts=True)
    #models_no_dropouts = get_all_model_iterations(dropouts=False)
    #hist_wi_dropouts = test_accuracy_for_model_iterations(models_wi_dropouts, data_loader)
    #hist_no_dropouts = test_accuracy_for_model_iterations(models_no_dropouts, data_loader)
    path_hist_all_models_no_dropouts = './model_states/history__pytk_rock_paper_all_models__scissors_Dropouts_False_ds_VAL'
    path_hist_all_models_wi_dropouts = './model_states/history__pytk_rock_paper_all_models__scissors_Dropouts_True_ds_VAL'
    #save_history(path_hist_all_models_no_dropouts, hist_no_dropouts)
    #save_history(path_hist_all_models_wi_dropouts, hist_wi_dropouts)
    hist_no_dropouts = load_history(path_hist_all_models_no_dropouts)
    hist_wi_dropouts = load_history(path_hist_all_models_wi_dropouts)
    show_training_accuracy(hist_no_dropouts, step_size=10)
    show_training_accuracy(hist_wi_dropouts, step_size=10)
    compare_histories(path_hist_all_models_no_dropouts, path_hist_all_models_wi_dropouts, compare_loss=False)
