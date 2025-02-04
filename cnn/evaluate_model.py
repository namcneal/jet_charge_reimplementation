from matplotlib import pyplot as plt
from math import floor
import numpy as np
import scipy as sci
import os
import random
import torch 

from Model import CNN
import mmap_ninja
from training import DatasetFromMemmap
from torch.utils.data import DataLoader

def main():
    model_dir = "./"
    model_filename = os.path.join("./training_logs/", "best_model.pth")
    base_data_dir = os.path.join("d:/", "up_down_2017_data")
    energy_gev = 1000
    kappa = 0.2

    down_quark_efficiency_roc(model_dir, model_filename, base_data_dir, energy_gev, kappa)


def down_quark_efficiency_roc(model_dir:str, model_filename:str, 
                              output_data_root_dir:str, energy_gev:int=1000, 
                              kappa:float=0.2):
    num_channels = 2
    model = CNN(num_channels)

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(0) # Set the GPU to use
        model = torch.nn.DataParallel(model) # Wrap the model with DataParallel
        model = model.to('cuda') # Move the model to the GPU

    state_dict = torch.load(os.path.join(model_dir, model_filename))

    model.load_state_dict(state_dict)
    model.eval()

    testing_images_folder = os.path.join(output_data_root_dir, "testing", "images")
    testing_labels_folder = os.path.join(output_data_root_dir, "testing", "labels")
    testing_dataset = DatasetFromMemmap(
        mmap_ninja.np_open_existing(testing_images_folder),
        mmap_ninja.np_open_existing(testing_labels_folder)
    )

    testing_dataloader = DataLoader(testing_dataset, batch_size=512, shuffle=False)
    
    down_quark_probs  = []
    down_quark_labels = []
    for (idx, (images, labels)) in enumerate(testing_dataloader):
        down_quark_labels.append(labels[:,1].detach().numpy())
        
        images = images.float()
        labels = labels.float()
        if use_gpu:
            images.to('cuda')
            labels.to('cuda')

        with torch.no_grad():
            output = model(images)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        down_quark_probs.append(probabilities[:,1].cpu().detach().numpy())

    down_quark_probs  = np.concatenate(down_quark_probs) 
    down_quark_labels = np.concatenate(down_quark_labels)
    print("Prob of down quark: ", down_quark_probs)

    thresholds  = np.linspace(0, 1, 500)
    down_quark_efficiencies = np.empty(len(thresholds))
    up_quark_efficiencies   = np.empty(len(thresholds))

    for t_idx, threshold in enumerate(thresholds):
        up_quark_labels = 1 - down_quark_labels
        num_down_quarks = np.sum(down_quark_labels)
        num_up_quarks   = np.sum(up_quark_labels)
        assert num_down_quarks + num_up_quarks == np.shape(down_quark_labels)[0]

        predictions_is_down = down_quark_probs > threshold
        predictions_is_up   = down_quark_probs <= threshold

        true_positives  = np.dot(predictions_is_down, down_quark_labels)
        true_negatives  = np.dot(predictions_is_up, up_quark_labels)

        down_quark_efficiencies[t_idx] = true_positives / num_down_quarks
        up_quark_efficiencies[t_idx]   = true_negatives / num_up_quarks

    # Sort the efficiencies along increasing horizontal axis, i.e. the down quark true positive rate
    increasing_down_quark_efficiencies = np.argsort(down_quark_efficiencies)
    down_quark_efficiencies = down_quark_efficiencies[increasing_down_quark_efficiencies]
    up_quark_efficiencies   = up_quark_efficiencies[increasing_down_quark_efficiencies]
    
    auc = sci.integrate.simpson(down_quark_efficiencies, up_quark_efficiencies)

    plt.plot(down_quark_efficiencies, up_quark_efficiencies, color='navy', lw=2)    
    plt.fill_between(down_quark_efficiencies, up_quark_efficiencies, color='navy', alpha=0.2)
    plt.text(0.6, 0.1, "AUC: {:.3f}".format(auc), fontsize=12)

    plt.grid()
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylabel("Up Quark (Background) Rejection")
    plt.xlabel("Down Quark (Signal) Acceptance")
    plt.title(r"ROC curve for {} GeV Jets (2017 Data) at $\kappa$={}".format(energy_gev, kappa))

    fig_filepath = os.path.join(output_data_root_dir, "roc_curve_for_{}_GeV_jets_at_kappa_{}_2017_data.png".format(energy_gev, kappa))
    plt.savefig(fig_filepath)
    plt.show()


