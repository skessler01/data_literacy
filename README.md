# Checking the Sky or the Screen: Disentangling Observed and Forecasted Weather Effects on Cycling Activity}
Project by Martin Gruber, Silja Keßler, Natalie Kraus und Johanna Mauch conducted as part of the course "Data Literacy" (WiSe 2025/26) by Prof. Philipp Hennig.

## Abstract
It is no surprise that cycling behavior is influenced by weather—but what is the more driving factor:
current observations or forecasts? We compare the ability of different models incorporating observed and forecasted weather to predict bicycle
counts from seven cities in Baden-Württemberg. Our results indicate that cyclists integrate both current and anticipated weather conditions into their decision-making, with a combined model providing the most accurate predictions; specifically, both factors predict a decline in cycling activity for lower temperatures and rainfall. These findings provide insight into psychological considerations underlying cycling decisions.
---

## Repository Structure
data/ :  Raw and processed datasets
exploration/ : Exploratory data analysis
figures/ : Figures generated for the paper
notebooks/ : Preprocessing, Analysis and evaluation notebooks
utils/ : Utility functions

---

## Data
Due to size, the datasets are not available on github 
but the project provides methods to download and 
generate all necessary data.

The main datasources are https://mobidata-bw.de/daten/eco-counter 
for bike count data and weather data from https://open-meteo.com/

Both datasets come as .csv files

The download of these files happens in notebooks/preprocessing_and_mstl.ipynb

---

## Exploration
This folder contains notebooks that **not part of our submitted project**. However, they contain exploratory analyses conducted to understand the data and inform analysis and evaluation.
Note that some of these notebooks can no longer be executed as they use an outdated dataset but still provide insights in the process leading to conceptual 
decisions for the main analysis. 
- cycling_promotion.ipynb, data_preprocessing.ipynb and locations.ipynb explore the initial (now outdated) bike dataset and potential problems
- decomposition_mstl.ipynb, iterative_decomposition.ipynb explore data decomposition techniques
- exploration.ipynb, exploration_visualizations.ipynb and weather_data_exploration.ipynb explore the raw dataset without and with weather combined
- weather_effect_exploration_GLMs.ipynb and gams.ipynb explore GLMs and GAMs for extracting weather effects
- outliers.ipynb explores techniques for filtering out outliers in a complex time series

---

## Notebooks
Main analysis and evaluation workflows.

- preprocessing_and_mstl.ipynb downloads/loads bike and weather data 
preprocesses it counter-wise by removing outliers and handling missing 
data. Then, MSTL decomposition is performed and the results are saved into 
data
- gams_pipeline.ipynb contains the GAM training pipeline including saving 
results (metrics, partial dependence curves)
- evaluation.ipynb evaluates and visualizes the GAM results
- paper_plots.ipynb contains the code for creating the plots shown in the paper
- other_plots.ipynb contains plots evaluating results from the analysis that are
not shown in the paper

---

## Figures
Figures used in the paper.

- bike_traffic_tuebingen.pdf: Figure 1 (heatmap) showing the hourly bike traffic recorded by a single counter at Unterführung Steinlach in Tübingen 
- forest_plot_mae_differences.pdf: Figure 2 (forest plot) showing the pairwise comparison of model predictive performance
- pred_vs_obs_wheather.pdf: Figure 3 (line plot + histogram) showing change in bike counts depending on temperature relative to the mean temperature

---

## Utils
Shared utility code.

- preprocessing_utils.py contains all util methods for 
downloading bike and weather data as well as the handling 
of missing data and outliers
- mstl_utils.py contains methods to perform mstl on the bike 
data and save the results
- gam_result_utils.py contains methods for computing and saving the 
partial dependence curves of the gams for weather and rain

---

## Reproducibility
Instructions for reproducing the results.

To perform the analysis, the notebooks need to be executed in the following order:
1. preprocessing_and_mstl.ipynb to load and preprocess data and perform seasonal decomposition using MSTL
2. gams_pipeline.ipynb to train Generalize additive models (GAMs)
3. evaluation.ipynb to evaluate GAMs
4. paper_plots.ipynb/other_plots.ipynb to reproduce plots (used in the paper)


