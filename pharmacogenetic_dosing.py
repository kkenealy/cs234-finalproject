import numpy as np
import pandas as pd

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

def make_one_hot_vector_vkorci_unknown(row, col_name):
    if row[col_name] != "A/A" or row[col_name] != "A/G" or row[col_name] != "G/G":
        return 1.
    return 0


def calculate_weekly_pharmacogenetic_dosing(row):
    dose = 5.6044
    dose -= 0.2614 * row['Age in Decades']
    dose += 0.0087 * row['Height (cm)']
    dose += 0.0128 * row['Weight (kg)']
    dose -= 0.8677 * row['VKORC1 A/G']
    dose -= 1.6974 * row['VKORC1 A/A']
    dose -= 0.4854 * row['VKORC1 Unknown']
    dose -= 0.5211 * row['CYP2C9 *1/*2']
    dose -= 0.9357 * row['CYP2C9 *1/*3']
    dose -= 1.0616 * row['CYP2C9 *2/*2']
    dose -= 1.9206 * row['CYP2C9 *2/*3']
    dose -= 2.3312 * row['CYP2C9 *3/*3']
    dose -= 0.2188 * row['CYP2C9 Unknown']
    dose -= 0.1092 * row['Asian Race']
    dose += 0.2760 * row['Black or African American']
    dose += 0.1032 * row['Mixed or Missing Race']
    dose += 1.1816 * row['Enzyme Inducer Status']
    dose -= 0.5503 * row['Amiodarone (Cordarone)']
    return dose ** 2

def calculate_dosage_matches(row):
    correct_dose = row["Therapeutic Dose of Warfarin"]
    prescribed_dose = row['Weekly Warfarin Pharmacogenetic Dosing']

    if correct_dose < 21 and prescribed_dose < 21:
        return 1
    elif (correct_dose >= 21 and correct_dose <= 49) and (prescribed_dose >= 21 and prescribed_dose <= 49):
        return 1
    elif correct_dose > 49 and prescribed_dose > 49:
        return 1
    return 0

def calculate_performance(data):
    performance = 0

    # Drop patients that are missing any required quantities
    data = data.drop(data[data["Age"] == "NA"].index)
    data = data.drop(data[pd.isna(data["Height (cm)"])].index)
    data = data.drop(data[pd.isna(data["Weight (kg)"])].index)
    data = data.drop(data[pd.isna(data["Carbamazepine (Tegretol)"])].index)
    data = data.drop(data[pd.isna(data["Phenytoin (Dilantin)"])].index)
    data = data.drop(data[pd.isna(data["Rifampin or Rifampicin"])].index)
    data = data.drop(data[pd.isna(data["Amiodarone (Cordarone)"])].index)

    total_patients = data.values.shape[0]

    # Define new columns that are necessary for clinical dosing
    data['Age in Decades'] = data.apply(lambda row: find_age_in_decades(row), axis = 1)
    data['VKORC1 A/G'] = data.apply(lambda row: make_one_hot_vector(row, "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T", "A/G"), axis = 1)
    data['VKORC1 A/A'] = data.apply(lambda row: make_one_hot_vector(row, "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T", "A/A"), axis = 1)
    data['VKORC1 Unknown'] = data.apply(lambda row: make_one_hot_vector_vkorci_unknown(row, "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"), axis = 1)

    data['CYP2C9 *1/*2'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", '*1/*2'), axis = 1)
    data['CYP2C9 *1/*3'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", '*1/*3'), axis = 1)
    data['CYP2C9 *2/*2'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", '*2/*2'), axis = 1)
    data['CYP2C9 *2/*3'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", '*2/*3'), axis = 1)
    data['CYP2C9 *3/*3'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", '*3/*3'), axis = 1)
    data['CYP2C9 Unknown'] = data.apply(lambda row: make_one_hot_vector(row, "Cyp2C9 genotypes", "NA"), axis = 1)

    data['Asian Race'] = data.apply(lambda row: identify_asian_race(row), axis = 1)
    data['Black or African American'] = data.apply(lambda row: identify_black_or_african_american(row), axis = 1)
    data['Mixed or Missing Race'] = data.apply(lambda row: identify_mixed_or_missing_race(row), axis = 1)
    data['Enzyme Inducer Status'] = data.apply(lambda row: determine_enzyme_inducer_status(row), axis = 1)

    # Compute weekly warfarin dosing based on clinical dosing strategy
    data['Weekly Warfarin Pharmacogenetic Dosing'] = data.apply(lambda row: calculate_weekly_pharmacogenetic_dosing(row), axis = 1)

    # Compute dosage matches
    data['Dosage Matches'] = data.apply(lambda row: calculate_dosage_matches(row), axis = 1)
    total_correct_patients = pd.DataFrame.sum(data['Dosage Matches'])

    performance = total_correct_patients/total_patients

    return performance
