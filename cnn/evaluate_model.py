from matplotlib import pyplot as plt
import numpy as np
import os
import torch 

from Model import CNN
from images_from_seed import output_group_file_name


def main():
    model_dir = "./"
    model_filename = "best_model.pth"
    data_dir = "D:\\image_data"
    energy_gev = 100
    kappa = 0.2

    down_quark_efficiency_roc(model_dir, model_filename, data_dir, energy_gev, kappa)


def down_quark_efficiency_roc(model_dir:str, model_filename:str, data_dir:str, energy_gev:int, kappa:float):
    num_channels = 2
    model = CNN(num_channels)

    # Remove the "module." prefix from the model's state_dict
    state_dict = torch.load(os.path.join(model_dir, model_filename))
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    TOTAL_NUM_DATA_GROUPS = 160
    NUM_TESTING_GROUPS    = 16
    testing_data_group_indices = range(TOTAL_NUM_DATA_GROUPS - NUM_TESTING_GROUPS, TOTAL_NUM_DATA_GROUPS)

    num_groups_to_load = 1
    selected_group_indices = testing_data_group_indices[-num_groups_to_load:]

    testing_images = np.concatenate(
        [np.load(output_group_file_name(data_dir, energy_gev, kappa, group, "image")) for group in selected_group_indices]
    )

    testing_labels = np.concatenate(
        [np.load(output_group_file_name(data_dir, energy_gev, kappa, group, "label")) for group in selected_group_indices]
    )    

    outputs = model(testing_images)
    
    down_quark_idx = 1
    prob_of_down_quark = outputs[:, down_quark_idx].detach().numpy()
    is_down_quark = testing_labels[:, down_quark_idx]

    thresholds  = np.linspace(0, 1, 1000)
    down_quark_efficiencies = np.empty(len(thresholds))
    up_quark_efficiencies   = np.empty(len(thresholds))

    for i, threshold in enumerate(thresholds):
        predictions = prob_of_down_quark > threshold

        true_positives  = np.sum(predictions * is_down_quark)
        false_positives = np.sum(predictions * (1 - is_down_quark))

        down_quark_efficiencies[i] = true_positives  / np.shape(is_down_quark)[0]
        up_quark_efficiencies[i]   = false_positives / np.shape(is_down_quark)[0]

    plot = plt.figure()
    plt.plot(up_quark_efficiencies, down_quark_efficiencies)
    plt.xlabel("False positive rate / Up quark efficiency")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.show()


