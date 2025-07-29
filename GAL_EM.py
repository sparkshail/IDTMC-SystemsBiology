from statistics import mean
from statistics import stdev
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
import math
import random

# read data
query_results = pd.read_csv("DTMC_GAL_Results_Size2.csv")
original = query_results[query_results["Model #"] == "OG"].copy()
three_intervals = query_results[1:4]
five_intervals = query_results[4:9]
seven_intervals = query_results[9:16]


# return distribution stats
def get_stats(result):
    mu = mean(result)
    sd = stdev(result)
    return [mu,sd]


# construct gaussian mixture using the EM algorithm and plot the distribution
def EM(values,num,n,line_colors):
    gmm = GaussianMixture(tol=0.000001)
    gmm.fit(np.expand_dims(values, 1))
    Gaussian_nr = 1

    x = np.linspace(min(values), max(values))  # plot the data

    for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_):
        print('Gaussian {:}: μ = {}, σ = {:.2}, weight = {:.2}'.format(n, mu, sd, p))
        g_s = stats.norm(mu, sd).pdf(x) * p
        plt.plot(x, g_s, line_colors[num-1],label="Distribution " + str(n))
        Gaussian_nr += 1
        sns.distplot(values, bins=7, kde=False, norm_hist=True, label="{} Intervals Results".format(n))
        plt.legend()
        return [mu,sd,p]


# calculate confidence interval from EM results

def confidence_interval(mean,sd,n):
    conf_int = stats.norm.interval(0.95, mean, sd/math.sqrt(n))
    print(conf_int)
    return conf_int


# calculates the smallest confidence interval, i.e. the most precise result
def interval_difference(intervals):
    print(intervals)
    min = intervals[0][1] - intervals[0][0]
    print("Interval distance", min)
    min_index = 0

    for i in range(1,len(intervals)):
        cur_diff = intervals[i][1] - intervals[i][0]
        print("Interval distance", cur_diff)
        if cur_diff < min:
            min = cur_diff
            min_index = i + 1

    return [min_index,min]


# calculates the difference between each EM distribution mean and the original model result
def calc_difference(og, means):
    differences = []
    intervals = [3, 5, 7]

    for i in range(0, len(means)):
        diff = abs(og-means[i])[0]
        print("Difference from the original using {} intervals: {}".format(intervals[i], diff))
        differences.append(diff)

    return differences

def main(query_index, line_colors):
    ##########################################################################
    # get the needed data

    num_intervals = ["3", "5", "7"]
    num = 0
    cur_distribution = 1

    og_result = original[query_index]
    result_3 = three_intervals[query_index]
    result_5 = five_intervals[query_index]
    result_7 = seven_intervals[query_index]

    results = [result_3, result_5, result_7]
    all_data = []
    means = []
    intervals = []

    ##########################################################################
    # perform the EM algorithm and plot results

    for result in results:
        stat = get_stats(result)
        print('Input Gaussian {:}: μ = {}, σ = {:.2}'.format(num_intervals[num], stat[0], stat[1]))
        values = result.array
        all_data.extend(values)
        em = EM(values,cur_distribution,num_intervals[num],line_colors)
        cur_interval = confidence_interval(em[0],em[1],int(num_intervals[num]))
        intervals.append(cur_interval)
        means.append(em[0])
        num = num+1
        cur_distribution = cur_distribution + 1

    x = np.linspace(min(all_data), max(all_data))
    plt.legend()

    #plt.title("Gaussian Distribution for the Long Run Percentage of Protein RAF1/RKIP")
    plt.title("Query 3 Gaussian Distribution")
    plt.xlabel("Percentage Value")
    plt.ylabel("Density")
    plt.show()

    ##########################################################################
    # determine smallest confidence interval, and either plot or table
    print("Smallest Confidence Interval: ", interval_difference(intervals))
    print("")

    ##########################################################################
    # calc the difference between EM results and og result, either plot or table
    calc_difference(og_result.array, means)


random.seed(100)
line_colors = ["blue", "chocolate", "green"]

# enter desired query
main("Query 3 Result ", line_colors)