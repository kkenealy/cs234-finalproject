import numpy as np
import pandas as pd
import utils

import pandas as pd
import torch
import utils

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

def calculate_dosage_matches(row):
    correct_dose = row["Therapeutic Dose of Warfarin"]
    prescribed_dose = row['Weekly Warfarin Clinical Dosing']

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
    data['Age in Decades'] = data.apply(lambda row: utils.find_age_in_decades(row), axis = 1)
    data['Asian Race'] = data.apply(lambda row: utils.identify_asian_race(row), axis = 1)
    data['Black or African American'] = data.apply(lambda row: utils.identify_black_or_african_american(row), axis = 1)
    data['Mixed or Missing Race'] = data.apply(lambda row: utils.identify_mixed_or_missing_race(row), axis = 1)
    data['Enzyme Inducer Status'] = data.apply(lambda row: utils.determine_enzyme_inducer_status(row), axis = 1)

    # Compute weekly warfarin dosing based on clinical dosing strategy
    data['Weekly Warfarin Clinical Dosing'] = data.apply(lambda row: calculate_weekly_warfarin_dosing(row), axis = 1)

    # Compute dosage matches
    data['Dosage Matches'] = data.apply(lambda row: calculate_dosage_matches(row), axis = 1)
    total_correct_patients = pd.DataFrame.sum(data['Dosage Matches'])

    performance = total_correct_patients/total_patients

    return performance
