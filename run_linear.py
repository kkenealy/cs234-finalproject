import numpy as np
import pandas as pd
import torch
import utils

path = "/Users/kathleenkenealy/Downloads/cs234project/data/warfarin.csv"


class LinearUCB():
    def __init__(self, alpha, K, d):
        super(LinearUCB, self).__init__()
        self.alpha = alpha
        self.K = K
        self.d = d

        self.A_low = torch.eye(d, dtype=torch.float64)
        self.A_mid = torch.eye(d, dtype=torch.float64)
        self.A_high = torch.eye(d, dtype=torch.float64)

        self.b_low = torch.zeros((d, 1), dtype=torch.float64)
        self.b_mid = torch.zeros((d, 1), dtype=torch.float64)
        self.b_high = torch.zeros((d, 1), dtype=torch.float64)

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
        print("Using", data.shape[1], "features for linear.")
        self.A_low = torch.eye(self.d, dtype=torch.float64)
        self.A_mid = torch.eye(self.d, dtype=torch.float64)
        self.A_high = torch.eye(self.d, dtype=torch.float64)

        self.b_low = torch.zeros((self.d, 1), dtype=torch.float64)
        self.b_mid = torch.zeros((self.d, 1), dtype=torch.float64)
        self.b_high = torch.zeros((self.d, 1), dtype=torch.float64)

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
            # print(correct_arm)

            theta_low = torch.mm(torch.inverse(self.A_low), self.b_low)
            theta_mid = torch.mm(torch.inverse(self.A_mid), self.b_mid)
            theta_high = torch.mm(torch.inverse(self.A_high), self.b_high)

            p_low = (torch.mm(torch.t(theta_low), x_t) + self.alpha * torch.sqrt(torch.mm(torch.t(x_t), torch.mm(torch.inverse(self.A_low), x_t)))).item()
            p_mid = (torch.mm(torch.t(theta_mid), x_t) + self.alpha * torch.sqrt(torch.mm(torch.t(x_t), torch.mm(torch.inverse(self.A_mid), x_t)))).item()
            p_high = (torch.mm(torch.t(theta_high), x_t) + self.alpha * torch.sqrt(torch.mm(torch.t(x_t), torch.mm(torch.inverse(self.A_high), x_t)))).item()
            # print(p_low, p_mid, p_high)

            p = torch.Tensor([p_low, p_mid, p_high])
            a_t = torch.argmax(p).item()
            # print(a_t)

            r_t = self.calculate_reward(a_t, correct_arm)
            if r_t == 0:
                num_correct += 1
            else:
                regret += 1
            fraction_incorrect[i] = (i + 1 - num_correct)/(i + 1)
            regret_array[i] = regret

            if a_t == 0:
                self.A_low = self.A_low + torch.mm(x_t, torch.t(x_t))
                self.b_low = self.b_low + x_t * r_t
            elif a_t == 1:
                self.A_mid = self.A_mid + torch.mm(x_t, torch.t(x_t))
                self.b_mid = self.b_mid + x_t * r_t
            else:
                self.A_high = self.A_high + torch.mm(x_t, torch.t(x_t))
                self.b_high = self.b_high + x_t * r_t

        performance = num_correct/num_patients
        print("Got ", performance, "% correct.")
        return performance, fraction_incorrect, regret_array



def linear_choose_features(data):
    """
    params: data, a pandas dataframe of size (# of patients, # of all features)

    returns: data, a torch tensor of size (# of patients, # of selected features) with new selected features
    """
    # Define new columns that are necessary for dosing
    print("Starting data processing.")
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
    column_list = ['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; A/A',
    'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; A/G',
    'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T; G/G',
    'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; T/T',
    'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; G/T',
    'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C; G/G',
    'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; C/C',
    'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; C/T',
    'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G; T/T',
    'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; C/C',
    'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; C/G',
    'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G; G/G',
    'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; A/A',
    'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; A/G',
    'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G; G/G',
    'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; C/C',
    'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; C/T',
    'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G; T/T',
    'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; A/A',
    'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; A/C',
    'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C; C/C',
    'Age in Decades',
    'Height (cm)',
    'Weight (kg)',
    'Asian Race',
    'Black or African American',
    'Mixed or Missing Race',
    'Enzyme Inducer Status',
    'Amiodarone (Cordarone)']

    # Convert to torch tensors
    final_data = torch.tensor(data[column_list].values)
    print("Finished with data processing. Data has ", data.shape[1], " features.")
    return final_data, correct_arms

def main():
    data = pd.read_csv(path)

    # Remove patients whose correct dosage is unknown
    data = data.drop(data[pd.isna(data["Therapeutic Dose of Warfarin"])].index)

    #Choose features, convert to torch tensor
    data, correct_arms = linear_choose_features(data)

    num_patients = data.shape[0]
    K = 3 # K is the number of actions
    d = data.shape[1] # d is the number of features
    alpha = 0.25

    performance = 0
    incorrect_performance = torch.zeros(10, num_patients)
    regret = torch.zeros(10, num_patients)
    for i in range(10):
        linucb = LinearUCB(alpha, K, d)
        performance_metric, fraction_incorrect, regret_array = linucb.train(data,correct_arms)
        incorrect_performance[i] = fraction_incorrect
        regret[i] = regret_array
        performance += performance_metric/10
    print("Average Performance: ", performance)



if __name__ == '__main__':
    main()
