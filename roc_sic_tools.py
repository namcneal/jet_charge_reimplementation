import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

def down_quark_sic_curves(model_dir:str, model_filename:str,
                        output_data_root_dir:str, energy_gev:int=1000, 
                        kappa:float=0.2):
    pass 

def down_quark_efficiency_roc(data_year:int, energy_gev:float, kappa:float,
                                probability_is_down_quark:np.ndarray,
                                down_quark_truth_labels:np.ndarray,
                                plot_output_dir:str, 
                                plot_output_filename:str,):
    # The labels should be just for the down quarks specifically
    # Check to make sure the passed array is just one dimensional
    if len(np.shape(down_quark_truth_labels)) > 1:
        raise ValueError("Down quark truth labels are not one dimensional.")

    thresholds  = np.linspace(0, 1, 500)
    down_quark_efficiencies = np.empty(len(thresholds))
    up_quark_efficiencies   = np.empty(len(thresholds))

    for t_idx, threshold in enumerate(thresholds):
        up_quark_labels = 1 - down_quark_truth_labels
        num_down_quarks = np.sum(down_quark_truth_labels)
        num_up_quarks   = np.sum(up_quark_labels)

        print("Arrays: ", up_quark_labels)
        print("Down quarks: ", num_down_quarks)
        print("Up quarks: ", num_up_quarks)        
        try:
            assert num_down_quarks + num_up_quarks == np.shape(down_quark_truth_labels)[0]
        except AssertionError:
            print("Down quark and up quark labels do not sum to the total number of labels.")
            print("Down quark labels: ", num_down_quarks)
            print("Up quark labels: ", num_up_quarks)
            print("Total labels: ", np.shape(down_quark_truth_labels)[0])
            return

        predictions_is_down = probability_is_down_quark > threshold
        predictions_is_up   = probability_is_down_quark <= threshold

        true_positives  = np.dot(predictions_is_down, down_quark_truth_labels)
        true_negatives  = np.dot(predictions_is_up, up_quark_labels)

        down_quark_efficiencies[t_idx] = true_positives / num_down_quarks
        up_quark_efficiencies[t_idx]   = true_negatives / num_up_quarks

    # Sort the efficiencies along increasing horizontal axis, i.e. the down quark true positive rate
    increasing_down_quark_efficiencies = np.argsort(down_quark_efficiencies)
    down_quark_efficiencies = down_quark_efficiencies[increasing_down_quark_efficiencies]
    up_quark_efficiencies   = up_quark_efficiencies[increasing_down_quark_efficiencies]
    
    auc = sp.integrate.simpson(up_quark_efficiencies, x=down_quark_efficiencies)

    plt.plot(down_quark_efficiencies, up_quark_efficiencies, color='navy', lw=2)    
    plt.fill_between(down_quark_efficiencies, up_quark_efficiencies, color='navy', alpha=0.2)
    plt.text(0.6, 0.1, "AUC: {:.3f}".format(auc), fontsize=12)

    plt.grid()
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylabel("Up Quark (Background) Rejection")
    plt.xlabel("Down Quark (Signal) Acceptance")
    plt.title(r"ROC curve for {} GeV Jets ({} Data) at $\kappa$={}".format(data_year, energy_gev, kappa))

    if plot_output_dir[-1] != '/':
        plot_output_dir += '/'
    fig_filepath = os.path.join(plot_output_dir, plot_output_filename)

    plt.savefig(fig_filepath)
    plt.show()

