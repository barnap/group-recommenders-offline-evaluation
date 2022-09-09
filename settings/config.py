dataset_folder = "ml-1m"
preprocessed_dataset_folder = "preprocessed_dataset"

# Preprocessing
min_ratings_per_user = 10
min_ratings_per_item = 10

# Group generation settings
group_sizes_to_test = [2, 3, 4, 5] # [2, 3, 4, 5, 6, 7, 8]
group_similarity_to_test = ["RANDOM", "SIMILAR", "DIVERGENT"]
group_number = 10 # 1000
shared_ratings = 5

# Evaluation settings
individual_rs_strategy = "LENSKIT_CF_USER"  # the used strategy for individual RS, I am keeping it generic to allow comparing more Individual Rec Sys if implemented, in a single run)
aggregation_strategies = ["BASE", ]  # list of implemented aggregation strategies we want to test, these should also be implemented)
recommendations_number = 10  # number of recommended items
# recommendations_ordered = "ranking"  # sequence or ranking
individual_rs_validation_folds_k = 0  # used for the k-fold validation)
group_rs_evaluation_folds_k = 5 # 10
evaluation_strategy = "COUPLED"  # coupled/decoupled
metrics = ["NDCG"]  # list of implemented metrics to evaluate)
