import numpy as np
import pandas as pd
from sklearn import linear_model
import torch
import utils
import math

path = "/Users/kathleenkenealy/Downloads/cs234project/data/warfarin.csv"


class Lasso():
    def __init__(self, q, h, lambda_1, lambda_2, K, d):
        super(Lasso, self).__init__()
        self.q = q
        self.h = h
        self.lambda_1 = lambda_1
        self.lambda_2_0 = lambda_2

        self.K = K
        self.d = d

    def initialize_single_tau(self, max_t, tau, i):
        n = 0
        while(True):
            for j in range(self.q * (i - 1) + 1, self.q * i + 1):
                t = (((2 ** n) - 1) * self.K * self.q) + j
                if t <= max_t:
                    tau.add(t)
                else:
                    return tau
            n += 1
        return tau

    def initialize_tau(self, max_t):
        T_1 = set([])
        T_1 = self.initialize_single_tau(max_t, T_1, 1)

        T_2 = set([])
        T_2 = self.initialize_single_tau(max_t, T_2, 2)

        T_3 = set([])
        T_3 = self.initialize_single_tau(max_t, T_3, 3)

        return T_1, T_2, T_3

    def calculate_Y_hat(self, tau, lambda_hat, X_t, Y_t, x_t_t):
        beta_predictor = linear_model.Lasso(alpha=lambda_hat/2, max_iter=100000, fit_intercept=False)
        X_t_hat = torch.index_select(X_t, 0, torch.tensor(list(tau), dtype=torch.long) - 1)
        Y_t_hat = torch.index_select(Y_t, 0, torch.tensor(list(tau), dtype=torch.long) - 1)
        beta_predictor.fit(X_t_hat, Y_t_hat)
        beta = torch.unsqueeze(torch.tensor(beta_predictor.coef_), 1)
        Y_hat = torch.mm(x_t_t, beta).item()
        # print("Given sets", Y_t_hat, "we predicted Y_hat = ", Y_hat)
        return Y_hat

    def calculate_reward(self, prescribed_arm, correct_dose):
        if correct_dose < 21 and prescribed_arm == 1:
            # print("MATCH")
            return 0
        elif (correct_dose >= 21 and correct_dose <= 49) and prescribed_arm == 2:
            # print("MATCH")
            return 0
        elif correct_dose > 49 and prescribed_arm == 3:
            # print("MATCH")
            return 0
        return -1

    def calculate_arm(self, dose):
        if dose < 21:
            return 1
        elif (dose >= 21 and dose <= 49):
            return 2
        else:
            return 3

    def train(self, data, correct_arms):
        """
        params: data, a torch tensor of size (# of patients, # of features)
                correct_arms, a torch tensor of size (# of patients, )
        """
        print("Using", data.shape[1], "features for lasso.")
        # Shuffle patients
        num_patients = data.shape[0]
        num_correct = 0
        num_1 = 0
        num_2 = 0
        num_3 = 0
        self.patient_order = torch.randperm(num_patients)

        # Initialize all sets
        S_1 = set([])
        S_2 = set([])
        S_3 = set([])
        T_1, T_2, T_3 = self.initialize_tau(num_patients)
        lambda_2 = self.lambda_2_0
        X_t = torch.transpose(torch.unsqueeze(data[self.patient_order[0], :], 1), 0, 1)
        Y_t = torch.zeros(1, dtype=torch.float64) + self.calculate_arm(correct_arms[self.patient_order[0]].item())

        fraction_incorrect = torch.zeros((num_patients), dtype=torch.float64)
        regret_array = torch.zeros((num_patients), dtype=torch.float64)
        regret = 0
        for t in range(1, num_patients + 1):
            # Pull new patient
            patient_idx = self.patient_order[t - 1]
            x_t = torch.unsqueeze(data[patient_idx, :], 1)
            correct_arm = correct_arms[patient_idx].item()

            # Update sets
            T_1_t = T_1.intersection(range(1, t + 1))
            T_2_t = T_2.intersection(range(1, t + 1))
            T_3_t = T_3.intersection(range(1, t + 1))
            S_1 = S_1.union(T_1_t)
            S_2 = S_2.union(T_2_t)
            S_3 = S_3.union(T_3_t)

            # Determine pi_t
            pi_t = -1
            if t in T_1:
                pi_t = 1
            elif t in T_2:
                pi_t = 2
            elif t in T_3:
                pi_t = 3
            else:
                K_hat = set([])
                x_t_t = torch.transpose(x_t, 0, 1)

                Y_hat_1 = self.calculate_Y_hat(T_1_t, self.lambda_1, X_t, Y_t, x_t_t)
                Y_hat_2 = self.calculate_Y_hat(T_2_t, self.lambda_1, X_t, Y_t, x_t_t)
                Y_hat_3 = self.calculate_Y_hat(T_3_t, self.lambda_1, X_t, Y_t, x_t_t)
                Y_hat_S_1 = self.calculate_Y_hat(S_1, lambda_2, X_t, Y_t, x_t_t)
                Y_hat_S_2 = self.calculate_Y_hat(S_2, lambda_2, X_t, Y_t, x_t_t)
                Y_hat_S_3 = self.calculate_Y_hat(S_3, lambda_2, X_t, Y_t, x_t_t)

                max_Y_hat = max(Y_hat_1, Y_hat_2, Y_hat_3)
                if Y_hat_1 >= max_Y_hat - (self.h/2):
                    K_hat.add((Y_hat_S_1, 1))
                if Y_hat_2 >= max_Y_hat - (self.h/2):
                    K_hat.add((Y_hat_S_2, 2))
                if Y_hat_3 >= max_Y_hat - (self.h/2):
                    K_hat.add((Y_hat_S_3, 3))

                max_val = float("-inf")
                max_action = -1
                for val, action in K_hat:
                    if val > max_val:
                        max_val = val
                        max_action = action
                pi_t = max_action

            if pi_t == 1:
                num_1 += 1
                S_1.add(t)
            elif pi_t == 2:
                num_2 += 1
                S_2.add(t)
            else:
                num_3 += 1
                S_3.add(t)
            lambda_2 = self.lambda_2_0 * math.sqrt((math.log(t) + math.log(self.d))/t)

            r_t = self.calculate_reward(pi_t, correct_arm)
            if r_t == 0:
                num_correct += 1
            else:
                regret += 1
            fraction_incorrect[t - 1] = (t - num_correct)/t
            regret_array[t - 1] = regret

            if t == 1:
                Y_t[0] = r_t
            else:
                X_t = torch.cat((X_t, torch.transpose(x_t, 0, 1)), dim=0)
                Y_t = torch.cat((Y_t, torch.tensor([r_t], dtype=torch.float64)), dim=0)

        performance = num_correct/num_patients
        print("Got ", performance, "% correct.")
        return performance, fraction_incorrect, regret_array



def lasso_choose_features(data):
    """
    params: data, a pandas dataframe of size (# of patients, # of all features)

    returns: data, a torch tensor of size (# of patients, # of selected features) with new selected features
    """
    # Define new columns that are necessary for dosing, remove old columns
    data = utils.define_new_cols(data, add_meds=False)
    correct_arms = torch.tensor(data['Therapeutic Dose of Warfarin'].values)
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
        'VKORC1 2255 consensus','VKORC1 -4451 consensus', 'Therapeutic Dose of Warfarin'], axis=1)
    data = data.fillna(0)

    # final_data = torch.tensor(data[column_list].values)
    final_data = torch.tensor(data.values)
    return final_data, correct_arms

def main():
    data = pd.read_csv(path)
    # Remove patients whose correct dosage is unknown
    data = data.drop(data[pd.isna(data["Therapeutic Dose of Warfarin"])].index)

    #Choose features, convert to torch tensor
    data, correct_arms = lasso_choose_features(data)

    num_patients = data.shape[0]
    q = 1
    h = 5
    lambda_1 = 0.1
    lambda_2 = 0.1
    K = 3 # K is the number of actions
    d = data.shape[1] # d is the number of features

    # lasso = Lasso(q, h, lambda_1, lambda_2, K, d)
    # performance_metric = lasso.train(data, correct_arms)
    performance_metric = 0
    lasso_incorrect_performance = torch.zeros(10, num_patients)
    lasso_regret = torch.zeros(10, num_patients)
    for i in range(10):
        lasso = Lasso(q, h, lambda_1, lambda_2, K, d)
        result, fraction_incorrect, regret_array = lasso.train(data,correct_arms)
        lasso_incorrect_performance[i] = fraction_incorrect
        lasso_regret[i] = regret_array
        performance_metric += result/10
    print("Average Lasso Performance: ", performance_metric)



if __name__ == '__main__':
    main()
