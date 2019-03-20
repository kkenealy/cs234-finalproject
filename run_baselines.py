import numpy as np
import pandas as pd
import baselines.fixed_doses as fixed_doses
import baselines.clinical_dosing as clinical_dosing
import baselines.pharmacogenetic_dosing as pharmacogenetic_dosing

path = "/Users/kathleenkenealy/Downloads/cs234project/data/warfarin.csv"

def main():
    data = pd.read_csv(path)
    # Remove patients whose correct dosage is unknown
    data = data.drop(data[pd.isna(data["Therapeutic Dose of Warfarin"])].index)

    # Check baseline performances
    fixed_doses_performance = fixed_doses.calculate_performance(data)
    clinical_dosing_performance = clinical_dosing.calculate_performance(data)
    pharmacogenetic_dosing_performance = pharmacogenetic_dosing.calculate_performance(data)

    print("Fixed Doses Performance: ", fixed_doses_performance)
    print("Clinical Dosing Performance: ", clinical_dosing_performance)
    print("Pharmacogenetic Dosing Performance: ", pharmacogenetic_dosing_performance)

if __name__ == '__main__':
    main()
