# CBB 6340 - Problem Set 0

### Student Information

Name: Jinnie Sun

NetID: js4872

### Instructions for Running Scripts

This project's scripts are written in Python and can be run from the command line.

### Exercise Answers & Results

#### Exercise 1: Clinical Decision Support - Temperature Tester

##### 1b. Identify ambiguity in problem description

1. **Temperature unit**: The use of Celcius or Fahrenheit degree is not clarified.
2. **Inclusive/Exclusive range**: Whether should a temperature that is exactly 1 degree away from the normal temperature be considered healthy is not clarified.

##### 1c. Testing

Inputs:

```python
chicken_tester(42)
human_tester(42)
chicken_tester(43)
human_tester(35)
human_tester(98.6)
```

Results:

```
True
False
False
False
False
```

#### Exercise 2: Analyzing COVID-19 Case Data

##### 2a. Data Acquisition and Loading

Data source: The New York Times. (2021). Coronavirus (Covid-19) Data in the United States. Retrieved 08/30/2025, from https://github.com/nytimes/covid-19-data.

##### 2b. Visualization of New Cases

Limitation of my approach:

The difference of cases between adjacent days are calculated here as new cases each day, but the case numbers are not updated everyday in this dataset. Therefore, there are many days with 0 new cases which is not the real circumstance. Shown as below:

![image-20250830222210497](assets/image-20250830222210497.png)

##### 2c. Find Peak Case Dates

Commands:

```python
print(peak_case('Washington'))
print(peak_case('California'))
print(peak_case('Texas'))
print(peak_case('Florida'))
```

Results:

```python
2022-01-18
2022-01-10
2022-01-03
2022-01-04
```

which is consistent with the [figure](assets/image-20250830222210497.png) above. 

##### 2e. Examine individual states

![image-20250830223426043](assets/image-20250830223426043.png)

```python
             date    state  fips    cases  deaths  new_cases
25213  2021-06-04  Florida    12  2289332   36985   -40527.0
```

At 2021-06-04, there is a significant decrease in cases. 

Hypothesis: It's possible that the negative case count was a result of correcting for duplicate or erroneous entries in the previous case records.

#### Exercise 3: Analyzing Population Data

##### 3a. Load and Examine Data:

What columns are present in the dataset?

+ name
+ age
+ weight
+ eyecolor

How many rows (representing individuals) does the dataset contain?

+ 152361

##### 3b. Analyze Age Distribution

Statistics for the age column:

```python
mean         39.510528
std          24.152760
min           0.000748
max          99.991547
```

Histogram of age distribution:

![image-20250830233339784](assets/image-20250830233339784.png)

The role of the number of bins

+ The number of bins determines the number of intervals the data is sorted into. Too few bins can hide important information of data, while too many bins can result in large noises.

Comment on any outliers or patterns you observe in the age distribution.

+ The frequency of individuals decreases sharply after the age of 70.

##### 3c. Analyze Weight Distribution

Statistics for the weight column:

```python
mean         60.884134
std          18.411824
min           3.382084
max         100.435793
```

Histogram of age distribution:

![image-20250830235038421](assets/image-20250830235038421.png)

Comment on any outliers or patterns you observe in the age distribution.

+ There is a significant high number of people with a weight of 68.

##### 3d. Explore Relationships

![image-20250830235245066](assets/image-20250830235245066.png)

General relationship between weights and ages:

+ Weight increases linearly with age until approximately 22, after which it stabilizes.

Identify and name any individual whose data does not follow the general relationship observed. 

+ Anthony Freeman

  ```python
                  name   age  weight eyecolor
  537  Anthony Freeman  41.3    21.7    green
  ```

Process for identifying this outlier

+ From the scatter plot, it is observed that only this outlier has a weight less than 30 and an age larger than 40.

  ```python
  print(df[(df['weight']<30) & (df['age']>40)])
  ```



### Appendix: Code
