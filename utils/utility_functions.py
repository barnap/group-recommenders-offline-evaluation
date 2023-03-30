import numpy as np
import pandas as pd
import settings.config as cfg
from aggregation_strategies.aggregators import AggregationStrategy
from evaluation_metrics.metric_evaluators import MetricEvaluator

#pre-processing for inverse propensity weighting, 
#for more details visit https://dl.acm.org/doi/abs/10.1145/3240323.3240355
def calculate_inverse_propensity_score(ratings_df, train_df, propensity_gama):
    items = ratings_df["item"].unique()
    
    #failsafe if some of the items never appeared in train data
    propensity_per_item = pd.DataFrame(1.0, index=items, columns=["propensity_score"])
    
    n_i_star_vector = train_df.groupby("item")["rating"].count()
    P_ui_vector = n_i_star_vector**((propensity_gama+1)/2)
    propensity_per_item.loc[P_ui_vector.index,"propensity_score"] = P_ui_vector

        
    return propensity_per_item  
    
    
#pre-processing for inverse propensity weighting
#Calculating  per-user fixed term of 1/\sum_{i \in R_u}(1/P_{u,i}), 
#    where R_u is a list of items known by user u and P_{u,i} is their propensity score
def calculate_inverse_propensity_score_user_normalization(propensity_per_item, test_df):
    inverse_propensity = 1/propensity_per_item
    
    local_df = test_df.copy()
    local_df = local_df.join(inverse_propensity, on="item")
    
    per_user_normalization_term = 1/local_df.groupby("user")["propensity_score"].sum()
        
    return per_user_normalization_term 
    
    
# Run all group RS strategies for all defined groups       
def generate_group_recommendations_forall_groups(test_df, group_composition, recommendations_number):
    group_recommendations = dict()
    for group_id in group_composition:
        
#         print(datetime.now(), group_id)
        
        # extract group info
        group = group_composition[group_id]
        group_size = group['group_size']
        group_similarity = group['group_similarity']
        group_members = group['group_members']
            
        group_ratings = test_df.loc[test_df['user'].isin(group_members)]
        
        group_rec = dict()
        for aggregation_strategy in cfg.aggregation_strategies:
            agg = AggregationStrategy.getAggregator(aggregation_strategy)
            group_rec = {**group_rec, **agg.generate_group_recommendations_for_group(group_ratings, recommendations_number)}
        
        
        group_recommendations[group_id] = group_rec
        
    return group_recommendations 
    
    
# Creating per_user_group_choices for evaluation (baseline) for tourism dataset
def create_per_user_group_choices(gr_composition, gr_choices):
    per_user_group_choices = pd.DataFrame(columns = ["user", "item", "rank"])
    for group_id in gr_composition:
        group = gr_composition[group_id]
        for member_id in group['group_members']:
            first_choice = gr_choices.loc[(gr_choices['group_id'] == group_id) & 
                                          (gr_choices['rank'] == 1)]['item'].values[0]
            second_choice = gr_choices.loc[(gr_choices['group_id'] == group_id) & 
                                          (gr_choices['rank'] == 2)]['item'].values[0]
            new_row1 = pd.DataFrame([[member_id, first_choice, 1]], 
                                       columns = ["user", "item", "rank"])
            new_row2 = pd.DataFrame([[member_id, second_choice, 2]], 
                                       columns = ["user", "item", "rank"])
            per_user_group_choices = pd.concat([per_user_group_choices, new_row1, new_row2], 
                                               axis=0, ignore_index=True)
    
    return per_user_group_choices

# Creating user_satisfaction for evaluation (baseline) for tourism dataset
def create_per_user_satisfaction(gr_composition, gr_choices, user_feedback):
    user_satisfaction = pd.DataFrame(columns = ["user", "item", "rating"])
    for group_id in gr_composition:
        group = gr_composition[group_id]
        for member_id in group['group_members']:
            first_choice = gr_choices.loc[(gr_choices['group_id'] == group_id) & 
                                          (gr_choices['rank'] == 1)]['item'].values[0]
            second_choice = gr_choices.loc[(gr_choices['group_id'] == group_id) & 
                                          (gr_choices['rank'] == 2)]['item'].values[0]
            satisfaction = user_feedback.loc[user_feedback['user_id'] == member_id]['choice_sat'].values[0]
            new_row1 = pd.DataFrame([[member_id, first_choice, satisfaction]], 
                                       columns = ["user", "item", "rating"])
            new_row2 = pd.DataFrame([[member_id, second_choice, satisfaction]], 
                                       columns = ["user", "item", "rating"])
            user_satisfaction = pd.concat([user_satisfaction, new_row1, new_row2], 
                                               axis=0, ignore_index=True)
    
    return user_satisfaction
    
#For all lists of group recommendations, evaluate their quality w.r.t. all required metrics    
def evaluate_group_recommendations_forall_groups(
    ground_truth, 
    group_recommendations, 
    group_composition, 
    propensity_per_item,
    per_user_propensity_normalization_term, 
    current_fold,
    evaluation_ground_truth,            
    binarize_feedback_positive_threshold,
    binarize_feedback,
    feedback_polarity_debiasing):
#     group_evaluations = dict()
    evaluation_strategy = cfg.evaluation_strategy
    metrics = cfg.metrics
    group_evaluations = list()
    for group_id in group_composition:
        
        #print(datetime.now(), group_id)
        
        # extract group info
        group = group_composition[group_id]
        group_size = group['group_size']
        group_similarity = group['group_similarity']
        group_members = group['group_members']
        group_rec = group_recommendations[group_id]
            
        # filter ratings in ground_truth for the group members
        group_ground_truth = ground_truth.loc[ground_truth['user'].isin(group_members)]
        

        for aggregation_strategy in group_rec:
            agg_group_rec = group_rec[aggregation_strategy]
            agg_group_rec_eval = list()
            for metric in cfg.metrics:
    #             print(datetime.now(), aggregation_strategy)
                metric_evaluator = MetricEvaluator.getMetricEvaluator(metric)
#                 agg_group_rec_eval = {**agg_group_rec_eval, **metric_evaluator.evaluateGroupRecommendation(group_ground_truth, agg_group_rec, group_members)}
                agg_group_rec_eval = agg_group_rec_eval + metric_evaluator.evaluateGroupRecommendation(                                  
                                  group_ground_truth,
                                  agg_group_rec,
                                  group_members,
                                  propensity_per_item,
                                  per_user_propensity_normalization_term,
                                  evaluation_ground_truth,            
                                  binarize_feedback_positive_threshold,
                                  binarize_feedback,
                                  feedback_polarity_debiasing
                    )
    
            # Adding aggregation strategy info
            for row in agg_group_rec_eval:
                row['aggregation_strategy'] = aggregation_strategy
                row['group_id'] = group_id
                row['current_fold'] = current_fold
#             group_rec_eval[aggregation_strategy] = agg_group_rec_eval
        
            #print(agg_group_rec_eval)
            group_evaluations = group_evaluations + agg_group_rec_eval
        # Adding group_id info
#         group_evaluations[group_id] = group_rec_eval
        
    return group_evaluations    