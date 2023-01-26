from rock_paper_scissors_code_only import *
import torch

file_pattern_training_data_wo_hands = '/Users/amling/uni/shifumi/DataEng/datasets/combined_grey/combined/*/*.png'
file_pattern_training_data_wi_hands = '/Users/amling/uni/shifumi/DataEng/datasets/combined_pp_01_grey/combined/*/*.png'
file_pattern_validation_data_wo_hands = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-validation_set_grey/*/*.png'
file_pattern_validation_data_wi_hands = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-validation_set_pp_01_grey/*/*.png'
file_pattern_testing_data_wo_hands = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-testing_set_grey/*/*.png'
file_pattern_testing_data_wi_hands = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-testing_set_pp_01_grey/*/*.png'


def show_history(hist, save_path = None):
    val_loss_sp = None
    acc_sp = None
    if save_path:
        val_loss_sp = f'{save_path}_loss.png'
        acc_sp = f'{save_path}_accuracy.png'
    show_validation_loss(hist, save_path=val_loss_sp)
    show_training_accuracy(hist, save_path=acc_sp)


# first plot absolute accuracys in valitdation set next to each other
def compare_histories(hist_one, hist_two, hist_one_name, hist_two_name, compare_accuracy=True, compare_loss=True, save_path=None, step_size =1):
    # this compares two histories
    if compare_accuracy:
        for key in ['total', 'rock', 'paper', 'scissors']:
            sp = None
            if save_path:
                sp = f'{save_path}_acc_{key}.png'
            show_compare_training_accuracy([hist_one, hist_two],
                                           [hist_one_name, hist_two_name], key=key,
                                           colors=['b-', 'g-'], step_size=step_size,
                                           save_path=sp)
    if compare_loss:
        sp = None
        if save_path:
            sp = f'{save_path}_loss.png'
        show_compare_training_loss([hist_one, hist_two], [hist_one_name, hist_two_name],
                                   colors=['b-', 'g-'], step_size=step_size,
                                   save_path=sp)


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


def max_acc_from_history(hist):
    max = None
    index = None
    for i, x in enumerate(hist):
        if max is None:
            max = x
            index = i
        elif x['accuracy_measures']['total'] > max['accuracy_measures']['total']:
            max = x
            index = i
    return index, max


def experiment_DataEng_background_removal(output_dir='./experiment_DataEng/', generate_data=False,
                                          show_data_loader_images=False):
    # TODO fix the names on the plots
    training_history_no_preprocessing = './model_states/history__pytk_rock_paper_scissors_NOT_PREPROCESSED_100epoches__Dropouts_True__BatchNorm_False'
    training_history_wi_preprocessing = './model_states/history__pytk_rock_paper_scissors_100epoches__Dropouts_True__BatchNorm_False'
    hist_training_history_no_preprocessing = load_history(training_history_no_preprocessing)
    hist_training_history_wi_preprocessing = load_history(training_history_wi_preprocessing)
    # TODO implement save path
    show_history(hist_training_history_no_preprocessing, save_path=f'{output_dir}model_no_pp_complete_training')
    show_history(hist_training_history_wi_preprocessing, save_path=f'{output_dir}model_wi_pp_complete_training')
    compare_histories(hist_one=hist_training_history_no_preprocessing, hist_one_name='without hands detection',
                      hist_two=hist_training_history_wi_preprocessing, hist_two_name='with hands detection',
                      save_path=f'{output_dir}compare_models_complete_training_hist')

    model_trained_wo_hands_name = 'pytk_rock_paper_scissors_NOT_PREPROCESSED'
    model_trained_wi_hands_name = 'pytk_rock_paper_scissors'
    path_hist_all_models_no_hands_det = f'{output_dir}history__{model_trained_wo_hands_name}'
    path_hist_all_models_wi_hands_det = f'{output_dir}history__{model_trained_wi_hands_name}'
    models_NO_hands_detection = get_all_model_iterations(dropouts=True, model_name=model_trained_wo_hands_name)
    models_hands_detection = get_all_model_iterations(dropouts=True, model_name=model_trained_wi_hands_name)

    if generate_data or show_data_loader_images:
        # Load all data for comparison
        data_loader_training_data_wo_hands = load_data_from_pattern(file_pattern_training_data_wo_hands)
        data_loader_training_data_wi_hands = load_data_from_pattern(file_pattern_training_data_wi_hands)
        data_loader_validation_data_wo_hands = load_data_from_pattern(file_pattern_validation_data_wo_hands)
        data_loader_validation_data_wi_hands = load_data_from_pattern(file_pattern_validation_data_wi_hands)
        data_loader_testing_data_wo_hands = load_data_from_pattern(file_pattern_testing_data_wo_hands)
        data_loader_testing_data_wi_hands = load_data_from_pattern(file_pattern_testing_data_wi_hands)
        if show_data_loader_images:
            show_images_from_dataloader(data_loader_training_data_wo_hands)
            show_images_from_dataloader(data_loader_training_data_wi_hands)
            show_images_from_dataloader(data_loader_validation_data_wo_hands)
            show_images_from_dataloader(data_loader_validation_data_wi_hands)
            show_images_from_dataloader(data_loader_testing_data_wo_hands)
            show_images_from_dataloader(data_loader_testing_data_wi_hands)
    if generate_data:
        # generate all six histories train val test all images
        hist_wo_hands_train = test_accuracy_for_model_iterations(models_NO_hands_detection,
                                                                 data_loader_training_data_wo_hands)
        hist_wo_hands_val = test_accuracy_for_model_iterations(models_NO_hands_detection,
                                                               data_loader_validation_data_wo_hands)
        hist_wo_hands_test = test_accuracy_for_model_iterations(models_NO_hands_detection,
                                                                data_loader_testing_data_wo_hands)
        hist_wi_hands_train = test_accuracy_for_model_iterations(models_hands_detection,
                                                                 data_loader_training_data_wi_hands)
        hist_wi_hands_val = test_accuracy_for_model_iterations(models_hands_detection,
                                                               data_loader_validation_data_wi_hands)
        hist_wi_hands_test = test_accuracy_for_model_iterations(models_hands_detection,
                                                                data_loader_testing_data_wi_hands)
        # save all the 6 histories
        save_history(f'{path_hist_all_models_no_hands_det}_TRAIN', hist_wo_hands_train)
        save_history(f'{path_hist_all_models_no_hands_det}_VAL', hist_wo_hands_val)
        save_history(f'{path_hist_all_models_no_hands_det}_TEST', hist_wo_hands_test)
        save_history(f'{path_hist_all_models_wi_hands_det}_TRAIN', hist_wi_hands_train)
        save_history(f'{path_hist_all_models_wi_hands_det}_VAL', hist_wi_hands_val)
        save_history(f'{path_hist_all_models_wi_hands_det}_TEST', hist_wi_hands_test)
    else:
        hist_wo_hands_train = load_history(f'{path_hist_all_models_no_hands_det}_TRAIN')
        hist_wo_hands_val = load_history(f'{path_hist_all_models_no_hands_det}_VAL')
        hist_wo_hands_test = load_history(f'{path_hist_all_models_no_hands_det}_TEST')
        hist_wi_hands_train = load_history(f'{path_hist_all_models_wi_hands_det}_TRAIN')
        hist_wi_hands_val = load_history(f'{path_hist_all_models_wi_hands_det}_VAL')
        hist_wi_hands_test = load_history(f'{path_hist_all_models_wi_hands_det}_TEST')

    # hist_no_dropouts = load_history(path_hist_all_models_no_dropouts)
    # show the 6 accuracies
    show_training_accuracy(hist_wo_hands_train, step_size=10, save_path=f'{output_dir}model_no_pp__training')
    show_training_accuracy(hist_wo_hands_val, step_size=10, save_path=f'{output_dir}model_no_pp__val')
    show_training_accuracy(hist_wo_hands_test, step_size=10, save_path=f'{output_dir}model_no_pp__test')
    show_training_accuracy(hist_wi_hands_train, step_size=10, save_path=f'{output_dir}model_wi_pp__training')
    show_training_accuracy(hist_wi_hands_val, step_size=10, save_path=f'{output_dir}model_wi_pp__val')
    show_training_accuracy(hist_wi_hands_test, step_size=10, save_path=f'{output_dir}model_wi_pp__test')
    # compare 3 accuracies
    compare_histories(hist_one=hist_wo_hands_train, hist_one_name='without hands detection -- training',
                      hist_two=hist_wi_hands_train, hist_two_name='with hands detection -- training',
                      compare_loss=False, step_size=10,
                      save_path=f'{output_dir}model_comp_10steps__training')
    compare_histories(hist_one=hist_wo_hands_val, hist_one_name='without hands detection -- validation',
                      hist_two=hist_wi_hands_val, hist_two_name='with hands detection -- validation',
                      compare_loss=False, step_size=10,
                      save_path=f'{output_dir}model_comp_10steps__val')
    compare_histories(hist_one=hist_wo_hands_test, hist_one_name='without hands detection -- test',
                      hist_two=hist_wi_hands_test, hist_two_name='with hands detection -- test',
                      compare_loss=False, step_size=10,
                      save_path=f'{output_dir}model_comp_10steps__test')
    # testing set performance on unknown dataset for dif preprocessor
    if generate_data:
        hist_wo_hands_test_dif_ds = test_accuracy_for_model_iterations(models_NO_hands_detection,
                                                                       data_loader_testing_data_wi_hands)
        hist_wi_hands_test_dif_ds = test_accuracy_for_model_iterations(models_hands_detection,
                                                                       data_loader_testing_data_wo_hands)
        save_history(f'{path_hist_all_models_no_hands_det}_TEST_DIF_DS', hist_wo_hands_test_dif_ds)
        save_history(f'{path_hist_all_models_wi_hands_det}_TEST_DIF_DS', hist_wi_hands_test_dif_ds)
    else:
        hist_wo_hands_test_dif_ds = load_history(f'{path_hist_all_models_no_hands_det}_TEST_DIF_DS')
        hist_wi_hands_test_dif_ds = load_history(f'{path_hist_all_models_wi_hands_det}_TEST_DIF_DS')
    show_training_accuracy(hist_wo_hands_test_dif_ds, step_size=10, save_path=f'{output_dir}model_no_pp__training_dif_ds')
    show_training_accuracy(hist_wi_hands_test_dif_ds, step_size=10, save_path=f'{output_dir}model_wi_pp__training_dif_ds')
    compare_histories(hist_one=hist_wo_hands_test_dif_ds, hist_one_name='without hands detection -- dif test dataset',
                      hist_two=hist_wi_hands_test_dif_ds, hist_two_name='with hands detection -- dif test dataset',
                      compare_loss=False, step_size=10,
                      save_path=f'{output_dir}model_comp_10steps__training_dif_ds')
    # get the 2 accuracy plots and comparison of total


if __name__ == "__main__":
    # to see all implemented methods look at my experiment
    experiment_DataEng_background_removal()
