import numpy as np
import pandas as pd
import torch
import utils

class FixedDosage():
    def __init__(self):
        super(FixedDosage, self).__init__()

    def calculate_reward(self, prescribed_arm, correct_dose):
        if correct_dose < 21 and prescribed_arm == 0:
            # print("MATCH")
            return 0
        elif (correct_dose >= 21 and correct_dose <= 49) and prescribed_arm == 1:
            # print("MATCH")
            return 0
        elif correct_dose > 49 and prescribed_arm == 2:
            # print("MATCH")
            return 0
        return -1

    def train(self, data, correct_arms):
        """
        params: data, a torch tensor of size (# of patients, # of features)
                correct_arms, a torch tensor of size (# of patients, )
        """
        num_patients = data.shape[0]
        num_correct = 0
        patient_order = torch.randperm(num_patients)
        fraction_incorrect = torch.zeros((num_patients), dtype=torch.float64)
        regret_array = torch.zeros((num_patients), dtype=torch.float64)
        regret = 0
        for i in range(num_patients):
            # Pull new patient
            patient_idx = patient_order[i]
            x_t = torch.unsqueeze(data[patient_idx, :], 1)
            correct_arm = correct_arms[patient_idx].item()
            prescribed_arm = 1
            # print(correct_arm)

            r_t = self.calculate_reward(prescribed_arm, correct_arm)
            if r_t == 0:
                num_correct += 1
            else:
                regret += 1
            fraction_incorrect[i] = (i + 1 - num_correct)/(i + 1)
            regret_array[i] = regret

        performance = num_correct/num_patients
        print("Got ", performance, "% correct.")
        return performance, fraction_incorrect, regret_array

def calculate_performance(data):
    performance = 0
    total_patients = data.values.shape[0]

    # Calculate number of patients with medium_dosage
    data = data.drop(data[data["Therapeutic Dose of Warfarin"] < 21].index)
    medium_dosage = data.drop(data[data["Therapeutic Dose of Warfarin"] > 49].index)
    medium_dosage_patients = medium_dosage.values.shape[0]

    # Calculate performance
    performance = medium_dosage_patients/total_patients

    return performance
