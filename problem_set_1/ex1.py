import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

file = 'problem_set_1/data/pset1-patients.xml'
patients = ET.parse(file)
root = patients.getroot()

# -----1a. Plot Age Distribution-----
ages = []
for patient in root.find('patients').findall('patient'):
    age = patient.get('age')
    ages.append(float(age))

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