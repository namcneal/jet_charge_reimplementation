import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os


def down_quark_efficiency_roc_and_sic(data_year:int, energy_gev:float, kappa:float,
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
    
    if plot_output_dir[-1] != '/':
        plot_output_dir += '/'

    np.savez(os.path.join(plot_output_dir, os.path.join(plot_output_filename, ".npz")), 
            true_down_pos = down_quark_efficiencies,
            true_up_neg   = up_quark_efficiencies)


    ## ROC
    plt.clf()

    plt.plot(down_quark_efficiencies, up_quark_efficiencies, color='navy', lw=2)    
    plt.fill_between(down_quark_efficiencies, up_quark_efficiencies, color='navy', alpha=0.2)

    auc = sp.integrate.trapezoid(up_quark_efficiencies, x=down_quark_efficiencies)
    plt.text(0.6, 0.1, "AUC: {:.3f}".format(auc), fontsize=12)

    plt.grid()
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylabel("Up Quark (Background) Rejection")
    plt.xlabel("Down Quark (Signal) True Positive Rate")
    plt.title(r"ROC curve for {} GeV Jets ({} Data) at $\kappa$={}".format(data_year, energy_gev, kappa))

    fig_filepath = os.path.join(plot_output_dir, os.path.join(plot_output_filename, "ROC"))
    plt.savefig(fig_filepath)
    plt.clf()
    plt.close()

    ## SIC
    plt.clf()

    significance_improvement = down_quark_efficiencies / np.sqrt(up_quark_efficiencies)
    plt.plot(down_quark_efficiencies, significance_improvement, color='navy', lw=2)    

    plt.grid()
    plt.xticks(np.linspace(0,1,11))
    # plt.yticks(np.linspace(0,1,11))
    plt.ylabel("Significance Improvement")
    plt.xlabel("Down Quark (Signal) True Positive Rate")
    plt.title(r"SIC curve for {} GeV Jets ({} Data) at $\kappa$={}".format(data_year, energy_gev, kappa))

    fig_filepath = os.path.join(plot_output_dir, os.path.join(plot_output_filename, "SIC"))
    plt.savefig(fig_filepath)
    plt.clf()
    plt.close()
