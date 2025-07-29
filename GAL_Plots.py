import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", palette="dark")

# read data
query_results = pd.read_csv("DTMC_GAL_Results_Size2.csv")
original = query_results[query_results["Model #"] == "OG"].copy()
three_intervals = query_results[1:4]
five_intervals = query_results[4:9]
seven_intervals = query_results[9:16]

queries = ["Query 1 Result", "Query 2 Result", "Query 3 Result "]
ylabels = [""]


def plot(query, ylabel):
    # 3 intervals
    sns.lineplot(x=[1, 2, 3], y=three_intervals[query], label="3 Intervals", linewidth=2)
    # 5 intervals
    sns.lineplot(x=[1, 2, 3, 4, 5], y=five_intervals[query], label="5 Intervals", linewidth=2)
    # 7 intervals
    sns.lineplot(x=[1, 2, 3, 4, 5, 6, 7], y=seven_intervals[query], label="7 Intervals", linewidth=2)
    plt.legend()
    plt.xlabel("Submodel Number", fontsize="medium")
    plt.ylabel(ylabel, fontsize="medium")
    # Enter the desired plot titles here
    #plt.title("The probability that GAL7 eventually reaches a value of 1")
    #plt.title("The probability of reaching a state where GAL1=1")
    plt.title("Probability that Gal6=1 is within 3 steps of Gal1=0")
    plt.show()


for query in queries:
    plot(query,"Probability")


