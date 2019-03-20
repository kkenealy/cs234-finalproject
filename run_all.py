import numpy as np
import pandas as pd
import torch
import utils
from run_linear import LinearUCB
from run_linear import linear_choose_features
from run_lasso import Lasso
from run_lasso import lasso_choose_features
from baselines.fixed_doses import FixedDosage
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import sem, t
from scipy import mean

path = "/Users/kathleenkenealy/Downloads/cs234project/data/warfarin.csv"

def choose_features(data):
    """
    params: data, a pandas dataframe of size (# of patients, # of all features)

    returns: data, a torch tensor of size (# of patients, # of selected features) with new selected features
    """
    # Define new columns that are necessary for dosing
    data = utils.define_new_cols(data)
    correct_arms = torch.tensor(data['Therapeutic Dose of Warfarin'].values)

    # Select final columns
    data = data.drop(['PharmGKB Subject ID', 'Gender', 'Race', 'Ethnicity', 'Age',
        'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
        'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
        'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
        'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
        'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
        'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
        'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
        'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
        'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
        'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
        'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
        'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
        'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
        'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
        'Indication for Warfarin Treatment', 'Comorbidities', 'Medications',
        'Estimated Target INR Range Based on Indication', 'Cyp2C9 genotypes',
        'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3', 'Combined QC CYP2C9',
        'CYP2C9 consensus', 'VKORC1 -1639 consensus', 'VKORC1 497 consensus',
        'VKORC1 1173 consensus', 'VKORC1 1542 consensus','VKORC1 3730 consensus',
        'VKORC1 2255 consensus','VKORC1 -4451 consensus','Therapeutic Dose of Warfarin'], axis=1)
    data = data.fillna(0)

    # Convert to torch tensors
    final_data = torch.tensor(data.values)
    return final_data, correct_arms

def main():
    data = pd.read_csv(path)

    # Remove patients whose correct dosage is unknown
    data = data.drop(data[pd.isna(data["Therapeutic Dose of Warfarin"])].index)

    num_patients = data.shape[0]
    q = 1
    h = 5
    lambda_1 = 0.1
    lambda_2 = 0.1
    K = 3 # K is the number of actions
    alpha = 0.1

    # Get Fixed Metrics
    fixed_data, fixed_correct_arms = choose_features(data)
    d = fixed_data.shape[1] # d is the number of features
    performance_metric = 0
    fixed_incorrect_performance = torch.zeros(10, num_patients)
    fixed_regret = torch.zeros(10, num_patients)
    for i in range(10):
        fixed = FixedDosage()
        result, fraction_incorrect, regret_array = fixed.train(fixed_data, fixed_correct_arms)
        fixed_incorrect_performance[i] = fraction_incorrect
        fixed_regret[i] = regret_array
        performance_metric += result/10
    print("Average Fixed Performance: ", performance_metric)
    fixed_average_incorrect = torch.sum(fixed_incorrect_performance, 0)/10
    fixed_low_incorrect, fixed_high_incorrect = utils.compute_confidence_intervals(fixed_incorrect_performance)
    fixed_average_regret = torch.sum(fixed_regret, 0)/10
    fixed_low_regret, fixed_high_regret = utils.compute_confidence_intervals(fixed_regret)

    # Get Lasso Metrics
    lasso_data, lasso_correct_arms = lasso_choose_features(data)
    d = lasso_data.shape[1] # d is the number of features
    performance_metric = 0
    lasso_incorrect_performance = torch.zeros(10, num_patients)
    lasso_regret = torch.zeros(10, num_patients)
    for i in range(10):
        lasso = Lasso(q, h, lambda_1, lambda_2, K, d)
        result, fraction_incorrect, regret_array = lasso.train(lasso_data, lasso_correct_arms)
        lasso_incorrect_performance[i] = fraction_incorrect
        lasso_regret[i] = regret_array
        performance_metric += result/10
    print("Average Lasso Performance: ", performance_metric)
    lasso_average_incorrect = torch.sum(lasso_incorrect_performance, 0)/10
    lasso_low_incorrect, lasso_high_incorrect = utils.compute_confidence_intervals(lasso_incorrect_performance)
    lasso_average_regret = torch.sum(lasso_regret, 0)/10
    lasso_low_regret, lasso_high_regret = utils.compute_confidence_intervals(lasso_regret)

    # Get Linear Metrics
    linear_data, linear_correct_arms = linear_choose_features(data)
    d = linear_data.shape[1] # d is the number of features
    performance = 0
    linear_incorrect_performance = torch.zeros(10, num_patients)
    linear_regret = torch.zeros(10, num_patients)
    for i in range(10):
        linucb = LinearUCB(alpha, K, d)
        performance_metric, fraction_incorrect, regret_array = linucb.train(linear_data, linear_correct_arms)
        linear_incorrect_performance[i] = fraction_incorrect
        linear_regret[i] = regret_array
        performance += performance_metric/10
    print("Average Linear Performance: ", performance)
    linear_average_incorrect = torch.sum(linear_incorrect_performance, 0)/10
    linear_low_incorrect, linear_high_incorrect = utils.compute_confidence_intervals(linear_incorrect_performance)
    linear_average_regret = torch.sum(linear_regret, 0)/10
    linear_low_regret, linear_high_regret = utils.compute_confidence_intervals(linear_regret)


    # Create Plots

    matplotlib.rc('axes.spines', top = False, right = False)
    matplotlib.rc('axes', facecolor = 'white')
    # Create the plot object
    _, incorrect_graph = plt.subplots()
    incorrect_graph.plot(np.array(range(1, num_patients + 1)), np.array(linear_average_incorrect), lw = 1, color = '#539caf', alpha = 1)
    incorrect_graph.fill_between(np.array(range(1, num_patients + 1)), linear_low_incorrect, linear_high_incorrect, color = '#539caf', alpha = 0.4, label = 'Linear Bandit (95% CI)')

    incorrect_graph.plot(np.array(range(1, num_patients + 1)), np.array(lasso_average_incorrect), lw = 1, color = '#ffa500', alpha = 1)
    incorrect_graph.fill_between(np.array(range(1, num_patients + 1)), lasso_low_incorrect, lasso_high_incorrect, color = '#ffa500', alpha = 0.4, label = 'Lasso Bandit (95% CI)')

    incorrect_graph.plot(np.array(range(1, num_patients + 1)), np.array(fixed_average_incorrect), lw = 1, color = '#a500ff', alpha = 1)
    incorrect_graph.fill_between(np.array(range(1, num_patients + 1)), fixed_low_incorrect, fixed_high_incorrect, color = '#a500ff', alpha = 0.4, label = 'Fixed Dosage (95% CI)')

    incorrect_graph.set_title("Average Fraction of Incorrect Dosing Decisions")
    incorrect_graph.set_xlabel("Number of Patients Seen")
    incorrect_graph.set_ylabel("Fraction of Incorrect Descisions")
    incorrect_graph.legend(loc = 'best')


    _, regret_graph = plt.subplots()
    regret_graph.plot(np.array(range(1, num_patients + 1)), np.array(linear_average_regret), lw = 1, color = '#539caf', alpha = 1)
    regret_graph.fill_between(np.array(range(1, num_patients + 1)), linear_low_regret, linear_high_regret, color = '#539caf', alpha = 0.4, label = 'Linear Bandit (95% CI)')

    regret_graph.plot(np.array(range(1, num_patients + 1)), np.array(lasso_average_regret), lw = 1, color = '#ffa500', alpha = 1)
    regret_graph.fill_between(np.array(range(1, num_patients + 1)), lasso_low_regret, lasso_high_regret, color = '#ffa500', alpha = 0.4, label = 'Lasso Bandit (95% CI)')

    regret_graph.plot(np.array(range(1, num_patients + 1)), np.array(fixed_average_regret), lw = 1, color = '#a500ff', alpha = 1)
    regret_graph.fill_between(np.array(range(1, num_patients + 1)), fixed_low_regret, fixed_high_regret, color = '#a500ff', alpha = 0.4, label = 'Fixed Dosage (95% CI)')

    regret_graph.set_title("Average Regret")
    regret_graph.set_xlabel("Number of Patients Seen")
    regret_graph.set_ylabel("Regret")
    regret_graph.legend(loc = 'best')

    plt.ylim(0, num_patients + 10)

    plt.show()



if __name__ == '__main__':
    main()
