# causalppi

Code for: "A Neural Method for Partial Identification of Treatment Effects with Complex Instruments".

### Folder structure

Our python code are stored by different dataset in the folders, we take `rct_data` as an example:

1. `synthetic_data_generation`: script for generating datasets
2. `run_utils`: train models, create confidence intervals
3. `experiments_u/main`: run specific experiments and store results
4. `plot.ipynb`: plot and summarize the results