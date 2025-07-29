# IDTMC-SystemsBiology
# Statistical Inference of Probabilistic Model Checking on a Galactose Regulatory Model
Takes an uncertain discrete time Markov Chain model (IDTMC) and constructs discrete time Markov chain (DTMC) submodels. All models constructed from the Galactose Regulatory Model are stored in the "Models" folder, but the user can also create their own for analysis. "GAL_Properties" holds the 3 queries used for Galactose Regulatory Model analysis, but the user can also create their own. 

### Dependencies
* Please utilize the following commands for dependency installations:
  * pip install pandas
  * pip install seaborn
  * pip install matplotlib 
  * pip install numpy
  * pip install sklearn
  * pip install scipy
  * pip install pymc 
  * pip install arviz
  
* You will also need to download the following applications:
  * Anaconda
  * PRISM Model Checker 4.8.1
    * Be sure the version matches your Python installation
  
### Directions
1. In GAL.py, replace "num_disjoint" with the desired number of disjoint subintervals. Also, replace the "transitions" list with the desired interval probability ranges. 
2. Run GAL.py. This file separates the transition intervals into the desired number of disjoint subintervals. Then, it randomly selects values from the disjoint subintervals and ensures that all probabilities within one transition sum to 1. The transitions of the newly constructed submodels are printed to the console. Information is printed line by line with the new probability intervals followed by the randomly selected probabilities.  
3. Using the model information from GAL.py, construct the corresponding PRISM model files. Write a PRISM properties file containing the desired queries to be evaluated. Evaluate the queries on all submodel files. Do the same for the original PRISM model (the model before uncertainty was introduced). Store the query results in a csv file in the same structure as "DTMC_GAL_Results.csv".  
4. Plot the query results using GAL_Plots.py. Be sure to replace the existing title with the desired plot title. 
5. Run the Expectation-Maximization (EM) algorithm using GAL_EM.py. Be sure to enter the desired query to run the algorithm on. This file prints the EM results to the console and also plots the resulting distribution.
6. Run Markov chain Monte Carlo (MCMC) using GAL_MCMC.py. Replace "queries" with the desired queries and "titles" with the desired plot titles. This file runs the MCMC algorithms and plots the posterior density estimation. Results are printed to the console, but can also be written to a csv file by uncommenting "df.to_csv("GAL_MCMC_results.csv")".
