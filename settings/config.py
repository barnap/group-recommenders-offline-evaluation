dataset_folder = "ml-1m"
preprocessed_dataset_folder = "preprocessed_dataset"

# Preprocessing
min_ratings_per_user = 10
min_ratings_per_item = 10

# Group generation settings
group_sizes_to_test = [2,4,8] # [2, 3, 4, 5, 6, 7, 8]
group_similarity_to_test = ["SIMILAR_ONE_DIVERGENT"] #["RANDOM", "SIMILAR", "DIVERGENT", "SIMILAR_ONE_DIVERGENT"]
group_number = 20 # 1000
shared_ratings = 5

# Evaluation settings
individual_rs_strategy = "LENSKIT_CF_ITEM"  # the used strategy for individual RS, I am keeping it generic to allow comparing more Individual Rec Sys if implemented, in a single run)
aggregation_strategies = ["BASE", "GFAR", "EPFuzzDA"]  # list of implemented aggregation strategies we want to test, these should also be implemented)
recommendations_number = 20  # number of recommended items
# recommendations_ordered = "ranking"  # sequence or ranking
individual_rs_validation_folds_k = 0  # used for the k-fold validation)
group_rs_evaluation_folds_k = 5 # 10

evaluation_strategy = "COUPLED"  # COUPLED / DECOUPLED evaluation type (see https://dl.acm.org/doi/10.1145/3511047.3537650)

inverse_propensity_debiasing = False #For COUPLED only: True / False whether to normalize feedback with self-normalized inverse propensity score (see https://dl.acm.org/doi/10.1145/3511047.3537650)
inverse_propensity_gamma = 0.1 #gamma parameter of the inverse propensity weighting. Larger values indicate more penalization for popular items

binarize_feedback = False #True / False whether to binarize user feedback for the evaluation
binarize_feedback_positive_threshold = 4.0 # if the feedback should be binarize, this denotes the minimal positive value

feedback_polarity_debiasing = 2.0 #polarity debiasing parameter c from https://dl.acm.org/doi/10.1145/3511047.3537650 usage: rating = max(0, rating+c)

metrics = ["NDCG","BINARY"]  # list of implemented metrics to evaluate)
