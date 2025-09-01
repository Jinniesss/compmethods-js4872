import pandas as pd

d_icd_diag = pd.read_csv('problem_set_0/D_ICD_DIAGNOSES.csv') # diag & diag_id
diag_icd = pd.read_csv('problem_set_0/DIAGNOSES_ICD.csv') # patient_id & diag_id
patients = pd.read_csv('problem_set_0/PATIENTS.csv') # patient_id & info

## 4a
print(sum(patients['gender']=='F'))
print(sum(patients['gender']=='M'))

## 4b
def get_subject_ids(diagnosis: str) -> list:
    diag_icd9 = d_icd_diag[(d_icd_diag['short_title']==diagnosis)|(d_icd_diag['long_title']==diagnosis)]['icd9_code'].iloc[0]
    sub_ids = diag_icd[diag_icd['icd9_code']==diag_icd9]['subject_id'].to_list()
    print(sub_ids)
    return sub_ids

get_subject_ids("Intestinal infection due to Clostridium difficile")

## 4c Function testing

def func_test(diagnosis: str):
    sub_ids = get_subject_ids(diagnosis)
    # 1. Check if all found subject_ids have the right diagnosis.
    sub_diag = diag_icd[diag_icd['subject_id'].isin(sub_ids)][['subject_id','icd9_code']]
    sub_diag['short_name'] = sub_diag['icd9_code'].map(d_icd_diag.set_index('icd9_code')['short_title'])
    sub_diag['long_name'] = sub_diag['icd9_code'].map(d_icd_diag.set_index('icd9_code')['long_title']) 
    sub_diag = sub_diag[(sub_diag['short_name']==diagnosis)|(sub_diag['long_name']==diagnosis)]
    print(sub_diag)
    for id in sub_ids:
        if id not in sub_diag['subject_id'].to_list():
            print(f"Error: subject_id {id} not found in diag_icd")
            return False
    print(f"All {len(sub_ids)} subject_ids have the diagnosis {diagnosis}.")
    return True

func_test("Intestinal infection due to Clostridium difficile")

## 4d. Age Calculation
def calculate_age(diagnosis: str):
    sub_ids = get_subject_ids(diagnosis)
    subjects = patients[patients['subject_id'].isin(sub_ids)][['subject_id','dob','dod']]
    subjects['dob'] = pd.to_datetime(subjects['dob'])
    subjects['dod'] = pd.to_datetime(subjects['dod'])
    
    # Set dob and dod to_pydatetime() and then calculate age in days
    ages = []
    for i in range(len(subjects)):
        sub = subjects.iloc[i]
        age = sub['dod'].to_pydatetime() - sub['dob'].to_pydatetime()
        ages.append(age.days)
    subjects['age_days'] = ages
    print(subjects)
    return subjects
    
calculate_age("Intestinal infection due to Clostridium difficile")
