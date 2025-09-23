import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

file = 'problem_set_1/data/pset1-patients.xml'
patients_et = ET.parse(file)
patients = patients_et.getroot().find('patients').findall('patient')

# -----1a. Plot Age Distribution-----
ages = []
for patient in patients:
    age = patient.get('age')
    ages.append(float(age))

plt.figure()
plt.hist(ages, bins=range(0, 101, 5), edgecolor='black')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.xticks(range(0, 101, 10))
plt.grid(axis='y', alpha=0.75)
plt.savefig('problem_set_1/figures/age_distribution.png')
# plt.show()

# Determine if any patients share the same exact age.
print(len(set(ages)), len(ages))

# -----1b. Plot Gender Distribution-----
genders = {}
for patient in patients:
    gender = patient.get('gender')
    # print(gender)
    if gender not in genders.keys():
        genders[gender] = 0
    genders[gender] += 1
gender_labels = list(genders.keys())
count_values = list(genders.values())
plt.figure()
plt.bar(gender_labels, count_values)
plt.title('Distribution of Genders')
plt.xlabel('Gender')
plt.ylabel('Number of Patients')
plt.savefig('problem_set_1/figures/gender_distribution.png')
# plt.show()

# -----1c. Sort Patients by Age-----
patients_sorted_by_age = sorted(patients, key=lambda x: float(x.get('age')))
eldest = patients_sorted_by_age[-1]
print(f"Eldest patient: {eldest.get('name')}, Age {eldest.get('age')}, Gender {eldest.get('gender')}")

# ----- 1e. Binary Search for Specific Age-----
def binary_search(patients, target_age):
    low = 0
    high = len(patients) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age == target_age:
            return patients[mid]
        elif mid_age < target_age:
            low = mid + 1
        else:
            high = mid - 1

    return None

patient_41d5 = binary_search(patients_sorted_by_age, 41.5)
print(f"Patient aged 41.5: {patient_41d5.get('name')}, Age {patient_41d5.get('age')}, Gender {patient_41d5.get('gender')}")

# -----1f. Count Patients Above a Certain Age-----
def count_above_age(patients, target_age):
    count = 0
    low = 0
    high = len(patients) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age == target_age:
            count += high - mid
            return count
        elif mid_age < target_age:
            low = mid + 1
        else:
            count += high - mid + 1
            high = mid - 1
        
    return count

count_above_41d5 = count_above_age(patients_sorted_by_age, 41.5)
print(f"Number of patients older than 41.5: {count_above_41d5}")
# validate with a linear scan
print(len([age for age in ages if age > 41.5]))

# -----1g. Function for Age Range Query-----
def count_in_age_range(patients, low_age, high_age):
    count = 0
    low = 0
    high = len(patients) - 1

    # Find the first patient >= low_age
    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age < low_age:
            low = mid + 1
        else:
            high = mid - 1

    start_index = low

    # Find the last patient < high_age
    low = 0
    high = len(patients) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age < high_age:
            low = mid + 1
        else:
            high = mid - 1

    end_index = high

    if start_index <= end_index:
        count = end_index - start_index + 1

    return count

# validation
print(count_in_age_range(patients_sorted_by_age, 0, 100), len(ages))
print(count_in_age_range(patients_sorted_by_age, 30, 41.5), len([age for age in ages if 30 <= age < 41.5]))

# -----1h. Function for Age and Gender Range Query-----
def count_male_in_age_range(patients, low_age, high_age):
    # data setup
    patients_male = [p for p in patients if p.get('gender') == 'male']
    
    count = count_in_age_range(patients, low_age, high_age)
    count_male = count_in_age_range(patients_male, low_age, high_age)

    return count, count_male

# validation
print(count_male_in_age_range(patients_sorted_by_age, 0, 100), (len(ages), len([p for p in patients if p.get('gender') == 'male' and 0 <= float(p.get('age')) < 100])))
print(count_male_in_age_range(patients_sorted_by_age, 30, 41.5), (len([age for age in ages if 30 <= age < 41.5]), len([p for p in patients if p.get('gender') == 'male' and 30 <= float(p.get('age')) < 41.5])))