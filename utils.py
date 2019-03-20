import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import sem, t
from scipy import mean
import torch
import math
from  collections import Counter

"""
Find the age in decades, as specified in Appendix 1f of the default project write-up.
"""
def find_age_in_decades(row):
    if row['Age'] == '0 - 9':
        return 0
    elif row['Age'] == '10 - 19':
        return 1
    elif row['Age'] == '20 - 29':
        return 2
    elif row['Age'] == '30 - 39':
        return 3
    elif row['Age'] == '40 - 49':
        return 4
    elif row['Age'] == '50 - 59':
        return 5
    elif row['Age'] == '60 - 69':
        return 6
    elif row['Age'] == '70 - 79':
        return 7
    elif row['Age'] == '80 - 89':
        return 8
    elif row['Age'] == '90+':
        return 9
    else: # If Age is NA
        return 0

def identify_asian_race(row):
    if row['Race'] == 'Asian':
        return 1
    return 0

def identify_black_or_african_american(row):
    if row['Race'] == 'Black or African American':
        return 1
    return 0

def identify_mixed_or_missing_race(row):
    if row['Race'] == 'Unknown':
        return 1
    return 0

def determine_enzyme_inducer_status(row):
    if row['Carbamazepine (Tegretol)'] == 1 or row['Phenytoin (Dilantin)'] == 1 or row['Rifampin or Rifampicin'] == 1:
        return 1
    return 0

def make_one_hot_vector(row, col_name, val):
    if row[col_name] == val:
        return 1.
    return 0

def make_medication_one_hot_vector(row, col_name, val):
    # print(row[col_name])
    opposite_val = "not " + val
    if isinstance(row[col_name], str):
        not_val = "not " + val
        no_val = "no " + val
        if not_val in row[col_name].lower():
            return 0
        if no_val in row[col_name].lower():
            return 0
        elif val in row[col_name].lower():
            return 1.
    return 0

def calculate_weekly_warfarin_dosing(row):
    dose = 4.0376
    dose -= 0.2546 * row['Age in Decades']
    dose += 0.0118 * row['Height (cm)']
    dose += 0.0134 * row['Weight (kg)']
    dose -= 0.6752 * row['Asian Race']
    dose += 0.4060 * row['Black or African American']
    dose += 0.0443 * row['Mixed or Missing Race']
    dose += 1.2799 * row['Enzyme Inducer Status']
    dose -= 0.5695 * row['Amiodarone (Cordarone)']
    return dose ** 2

def define_new_cols(data, add_meds=False):
    data['Age in Decades'] = data.apply(lambda row: find_age_in_decades(row), axis = 1)
    data['White Race'] = data.apply(lambda row: make_one_hot_vector(row, 'Race', 'White'), axis =1)
    data['Hispanic Race'] = data.apply(lambda row: make_one_hot_vector(row, 'Ethnicity', 'Hispanic or Latino'), axis =1)
    data['Not Hispanic Race'] = data.apply(lambda row: make_one_hot_vector(row, 'Ethnicity', 'not Hispanic or Latino'), axis =1)
    data['Asian Race'] = data.apply(lambda row: identify_asian_race(row), axis = 1)
    data['Black or African American'] = data.apply(lambda row: identify_black_or_african_american(row), axis = 1)
    data['Mixed or Missing Race'] = data.apply(lambda row: identify_mixed_or_missing_race(row), axis = 1)
    data['Enzyme Inducer Status'] = data.apply(lambda row: determine_enzyme_inducer_status(row), axis = 1)

    data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; A/A'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'A/A'), axis =1)
    data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; A/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'A/G'), axis =1)
    data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; G/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'G/G'), axis =1)

    data['VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; T/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'T/T'), axis =1)
    data['VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; G/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'G/T'), axis =1)
    data['VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; G/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'G/G'), axis =1)

    data['VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; C/C'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'C/C'), axis =1)
    data['VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; C/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'C/T'), axis =1)
    data['VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; T/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'T/T'), axis =1)

    data['VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; C/C'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'C/C'), axis =1)
    data['VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; C/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'C/G'), axis =1)
    data['VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; G/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'G/G'), axis =1)

    data['VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; A/A'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'A/A'), axis =1)
    data['VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; A/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'A/G'), axis =1)
    data['VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; G/G'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'G/G'), axis =1)

    data['VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; C/C'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'C/C'), axis =1)
    data['VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; C/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'C/T'), axis =1)
    data['VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; T/T'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'T/T'), axis =1)

    data['VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; A/A'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'A/A'), axis =1)
    data['VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; A/C'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'A/C'), axis =1)
    data['VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; C/C'] = data.apply(lambda row: make_one_hot_vector(row, 'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'C/C'), axis =1)

    data['Male'] = data.apply(lambda row: make_one_hot_vector(row, 'Gender', 'male'), axis = 1)
    data['Female'] = data.apply(lambda row: make_one_hot_vector(row, 'Gender', 'female'), axis = 1)

    data['Indication for Warfarin Treatment; 1'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '1'), axis = 1)
    data['Indication for Warfarin Treatment; 2'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '2'), axis = 1)
    data['Indication for Warfarin Treatment; 3'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '3'), axis = 1)
    data['Indication for Warfarin Treatment; 4'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '4'), axis = 1)
    data['Indication for Warfarin Treatment; 5'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '5'), axis = 1)
    data['Indication for Warfarin Treatment; 6'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '6'), axis = 1)
    data['Indication for Warfarin Treatment; 7'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '7'), axis = 1)
    data['Indication for Warfarin Treatment; 8'] = data.apply(lambda row: make_one_hot_vector(row, 'Indication for Warfarin Treatment', '8'), axis = 1)

    data['Comorbidities; No Cancer'] = data.apply(lambda row: make_one_hot_vector(row, 'Comorbidities', 'No Cancer'), axis = 1)
    data['Comorbidities; Cancer'] = data.apply(lambda row: make_one_hot_vector(row, 'Comorbidities', 'Cancer'), axis = 1)
    data['Comorbidities; NA'] = data.apply(lambda row: make_one_hot_vector(row, 'Comorbidities', 'NA'), axis = 1)

    if add_meds:
        set_of_meds = {}
        for i in range(data.shape[0]):
            med_string = data['Medications'][i]
            if isinstance(med_string, str):
                all_meds = med_string.lower().split("; ")
                # print(all_meds)
                for med in all_meds:
                    if med in set_of_meds:
                        set_of_meds[med] += 1
                    else:
                        set_of_meds[med] = 1

        count = Counter(set_of_meds)
        for med, v in count.most_common(20):
            column_title = "Medications; " + med
            data[column_title] = data.apply(lambda row: make_medication_one_hot_vector(row, 'Medications', med), axis=1)

    data['Bias'] = data.apply(lambda row: 1, axis = 1)
    # data['Weekly Warfarin Clinical Dosing'] = data.apply(lambda row: calculate_weekly_warfarin_dosing(row), axis = 1)

    return data

def plot_fraction_incorrect(x_data, y_data, x_label, y_label, title):
    matplotlib.rc('axes.spines', top = False, right = False)
    matplotlib.rc('axes', facecolor = 'white')

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

# Define a function for the line plot with intervals
def plot_fraction_incorrect_with_confidence(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title, set_equal=False, num_patients=5528):
    matplotlib.rc('axes.spines', top = False, right = False)
    matplotlib.rc('axes', facecolor = 'white')
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(np.array(x_data), np.array(y_data), lw = 1, color = '#539caf', alpha = 1)
    # Shade the confidence interval
    ax.fill_between(np.array(sorted_x), low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if set_equal:
        plt.ylim(0, num_patients + 10)

    # Display legend
    ax.legend(loc = 'best')
    plt.show()

def compute_confidence_intervals(data):
    n = 10
    confidence = 0.95
    # print(data.shape)
    mean = torch.sum(data, 0)/n
    # print(mean.shape)
    std_error = sem(data, axis=0)
    # print(std_error.shape)
    ppf = t.ppf((1 + confidence)/ 2, n - 1)
    # print(ppf)
    h = std_error * ppf
    return np.array(mean) - h, np.array(mean) + h
