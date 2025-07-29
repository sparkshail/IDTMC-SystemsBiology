import random
from matplotlib import pyplot as plt
import pymc as pm
import numpy as np
import pandas as pd
from scipy import stats
from statistics import mean
from statistics import stdev
import arviz as az
import math
log = np.log
pi = np.pi

# read data
query_results = pd.read_csv("DTMC_GAL_Results.csv")
original = query_results[query_results["Model #"] == "OG"].copy()
three_intervals = query_results[1:4]
five_intervals = query_results[4:9]
seven_intervals = query_results[9:16]
results = [three_intervals, five_intervals, seven_intervals]

means = []
sds = []
confidence_intervals = []
interval_sizes = []
distance_OG = []
query_list = []


def get_stats(result):
    mu = mean(result)
    sd = stdev(result)
    return [mu,sd]


def confidence_interval(mean,sd,n):
    conf_int = stats.norm.interval(0.95, mean, sd/math.sqrt(n))
    print(conf_int)
    return conf_int


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


def calc_difference(og, mean):
    diff = abs(og - mean)
    return diff


def mcmc(result, title):
    ##########################################################################

    stat = get_stats(result)
    print('Input Gaussian {:}: μ = {}, σ = {:.2}'.format("3", stat[0], stat[1]))


    with pm.Model() as model:
        prior = pm.Normal('mu', mu=stat[0], sigma=stat[1])
        step = pm.Metropolis()

        # sample with 6 independent Markov chains
        trace = pm.sample(draws=100, chains=6, step=step, return_inferencedata=True)

    print("trace", trace)
    summary = az.summary(trace, round_to=10)
    az.plot_posterior(trace)
    plt.title(title)
    plt.show()

    print(az.summary(trace, round_to=3))

    return summary


def main(query_index,res, title):
    # prepare data
    og_result = original[query_index]
    result = res[query_index]

    # run the MCMC
    summary = mcmc(result, title)

    cur_mean = summary["mean"][0]
    sd = summary["sd"][0]

    means.append(cur_mean)
    sds.append(sd)

    cur_int = confidence_interval(cur_mean,sd,3)
    confidence_intervals.append(cur_int)

    interval_sizes.append(cur_int[1] - cur_int[0])

    dif = calc_difference(og_result, cur_mean)
    distance_OG.append(dif[0])


random.seed(100)
queries = ["Query 1 Result", "Query 2 Result", "Query 3 Result "]
titles = ["Posterior Density Estimation of the probability that GAL7 eventually reaches a value of 1",
          "Posterior Density Estimation of the probability of reaching a state where GAL1=1",
          "Posterior Density Estimation of the probability that Gal6=1 is within 3 steps of Gal1=0"]

count = 0
for query in queries:
    for interval in results:
        try:
            if count < 3:
                title = titles[0]
            elif count < 6:
                title = titles[1]
            else:
                title = titles[2]
            query_list.append(query)
            main(query, interval, title)
        except:
            query_list.append("na")
            means.append("na")
            sds.append("na")
            confidence_intervals.append("na")
            distance_OG.append("na")
            interval_sizes.append("na")
        count = count + 1

data = {"Query": query_list, "mean": means, "standard_deviation": sds, "conf_interval": confidence_intervals,
        "distance_from_og": distance_OG, "int_size": interval_sizes}
df = pd.DataFrame(data)

# uncomment to write results
#df.to_csv("GAL_MCMC_results.csv")