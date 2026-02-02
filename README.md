# Project Title
Group Project Data Literacy (Johanna, Martin, Natalie, Silja)

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
Exploratory analyses conducted to understand the data and inform evaluation.
This folder contains notebooks exploring the data, techniques or features at 
all stages of the project as well as util functions used by them.
Some of these notebooks can no longer be executed due to changes in the 
dataset but still provide insights in the process leading to conceptual 
decisions for the main analysis. 
- cycling_promotion.ipynb, data_preprocessing.ipynb and locations.ipynb explored 
the default bike dataset and preprocessing steps
- decomposition_mstl.ipynb, iterative_decomposition.ipynb 
explore data decomposition techniques
- exploration.ipynb, exploration_visualizations.ipynb and weather_data_exploration.ipynb explore the unchanged dataset 
without and with weather combined
- weather_effect_exploration_GLMs.ipynb and gams.ipynb explore GLMs and GAMs 
for extracting weather effects
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

- Description of generated plots and tables

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

Execution Order: To perform the analysis, the notebooks need to be executed in the following order:
1. preprocessing_and_mstl.ipynb
2. gams_pipeline.ipynb
3. evaluation.ipynb
4. paper_plots.ipynb/other_plots.ipynb


