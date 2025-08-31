import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
data = pd.read_csv("problem_set_0/us-states.csv")

def new_cases(states):
    """
    Takes a list of state names and plots their new cases versus date using overlaid line graphs.
    """
    plt.figure()
    for state in states:
        state_data = data[data['state'] == state]
        state_data = state_data.sort_values('date')
        state_data['new_cases'] = state_data['cases'].diff().fillna(0)
        plt.plot(state_data['date'], state_data['new_cases'], label=state, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.title('New COVID-19 Cases Over Time')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10)) 
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # print(state_data[state_data['new_cases'] < 0])

new_cases(['Florida'])

def peak_case(state):
    """
    Takes the name of a state and returns the date of its highest number of new cases.
    """
    state_data = data[data['state'] == state]
    peak_date = state_data.loc[state_data['cases'].diff().idxmax()]['date']
    return peak_date

peak_case('Florida')