# RecSys 2022 tutorial: Off-line evaluation of group recommender systems
## Presenters: Francesco Barile, Amra Delic and Ladislav Peska

This repository contains supplementary materials for tutorial's hands-on session.
- Tourism and ML1M datasets
- Strategies to generate synthetic groups
- Group aggregation algorithms constructing group recommendations from individual preferences of group members
- Coupled and Decoupled evaluation protocols with basic debiasing options
- Several relevance-based and fairness evaluation metrics
- Results visualizations

## Requirements:
- Python 3.x, std. python scientific stack (numpy, pandas, scipy, matplotlib, seaborn), lenskit (for individual RS training and prediction only)

## Usage:
The evaluation pipeline comprises from the following steps: 
- pre-processing
- synthetic groups generation (ML1M only)
- training individual RS and predictions (ML1M only)
- generating groups recommendations
- evaluating generated recommendations based on selected eval. parameters (i.e., coupled/decoupled evaluation for ML1M, types of ground truth for Tourism)

Parameters of the evaluation pipeline are stored in settings/config.py file. Two default variants for ML1M and Tourism datasets are stored in config_movie_lens.py and config_tourism_dataset.py respectively.
The main evaluation pipeline is divided into several Python notebooks, specific for both datasets. These notebooks should be executed sequentially. 
- Tourism dataset: tourism_preprocessing.ipynb and grs_evaluation_final_tourism.ipynb
- ML1M dataset: ml1m_preprocessing.ipynb, groups_generation_final.ipynb and grs_evaluation_final_ml1m.ipynb

Note that results of each individual step of the evaluation pipeline are stored in the Pickles file, so they can be executed just once unless different parameters are selected.

- for ML1M dataset, several pre-computed evaluation runs are stored in preprocessed_dataset/evaluation_runs.zip If unziped in the same folder, one can continue directly to the results visualization cells in grs_evaluation_final_ml1m.ipynb and explore these pre-computed results.
