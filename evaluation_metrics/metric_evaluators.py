import numpy as np
import settings.config as cfg

# Evaluation Metrics strategies

from abc import ABC, abstractmethod


class MetricEvaluator(ABC):

    @staticmethod
    def getMetricEvaluator(metric):
        if metric == "NDCG":
            return NDCGEvaluator()
        elif metric == "DCG":
            return DCGEvaluator()
        elif metric == "BINARY":
            return BinaryEvaluator()
        elif metric == "BASE":
            return BaselinesEvaluators()
        return None

    @abstractmethod
    def evaluateGroupRecommendation(
            self,
            group_ground_truth,
            group_recommendation,
            group_members,
            propensity_per_item,
            per_user_propensity_normalization_term,
            evaluation_ground_truth,            
            binarize_feedback_positive_threshold,
            binarize_feedback,
            feedback_polarity_debiasing
    ):
        pass


class NDCGEvaluator(MetricEvaluator):

    def evaluateUserNDCG(self, user_ground_truth, group_recommendation, user_norm):
        # note that both dcg and idcg should be element-wise normalized via per_user_propensity_normalization_term
        # therefore, it can be excluded from calculations
        dcg = 0
        #         display(user_ground_truth)
        #         display(group_recommendation)
        for k, item in enumerate(group_recommendation):
            dcg = dcg + ((user_ground_truth.loc[
                              item, "final_rating"] if item in user_ground_truth.index else 0) / np.log2(k + 2))

        idcg = 0
        # what if intersection is empty?
        user_ground_truth.sort_values("final_rating", inplace=True, ascending=False)
        # print(user_ground_truth)
        # print(len(user_ground_truth),len(group_recommendation),min(len(user_ground_truth),len(group_recommendation)))
        for k in range(min(len(user_ground_truth), len(group_recommendation))):
            idcg = idcg + (user_ground_truth.iloc[k]["final_rating"] / np.log2(k + 2))
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0

        return ndcg, user_norm * dcg

    def evaluateGroupRecommendation(
            self,
            group_ground_truth,
            group_recommendation,
            group_members,
            propensity_per_item,
            per_user_propensity_normalization_term,
            evaluation_ground_truth,            
            binarize_feedback_positive_threshold,
            binarize_feedback,
            feedback_polarity_debiasing
    ):
        ndcg_list = list()
        dcg_list = list()
        for user in group_members:

            user_ground_truth = group_ground_truth.loc[group_ground_truth['user'] == user]
            user_ground_truth.set_index("item", inplace=True)

            # feedback binarization
            if evaluation_ground_truth != "GROUP_CHOICES":
                if binarize_feedback == True:
                    user_ground_truth["final_rating"] = 0
                    user_ground_truth.loc[user_ground_truth.rating >= binarize_feedback_positive_threshold,"final_rating"] = 1
                # basic polarity debiasing (max(0, rating + c))
                elif feedback_polarity_debiasing != 0.0:
                    user_ground_truth["final_rating"] = user_ground_truth.loc[:,"rating"] + feedback_polarity_debiasing
                    user_ground_truth.loc[user_ground_truth.final_rating < 0,"final_rating"] = 0              
                # no modifications to feedback
                else:
                    user_ground_truth["final_rating"] = user_ground_truth["rating"]
            else: 
                user_ground_truth["final_rating"] = 1
            
            #print(user_ground_truth.head(3))

                # self-normalized inverse propensity debiasing
            user_ground_truth.loc[:, "final_rating"] = user_ground_truth.loc[:, "final_rating"] / propensity_per_item[
                "propensity_score"]
            user_norm = 1.0
            if per_user_propensity_normalization_term is not None:
                user_norm = per_user_propensity_normalization_term[user]

            ndcg_user, dcg_user = self.evaluateUserNDCG(user_ground_truth, group_recommendation, user_norm)
            ndcg_list.append(ndcg_user)
            dcg_list.append(dcg_user)

            # failsafe for all negative results
            if np.amax(ndcg_list) > 0:
                ndcg_min_max = np.amin(ndcg_list) / np.amax(ndcg_list)
                dcg_min_max = np.amin(dcg_list) / np.amax(dcg_list)
            else:
                ndcg_min_max = 0.0
                dcg_min_max = 0.0
        return [
            {
                "metric": "NDCG",
                "aggr_metric": "mean",
                "value": np.mean(ndcg_list)
            },
            {
                "metric": "NDCG",
                "aggr_metric": "min",
                "value": np.amin(ndcg_list)
            },
            {
                "metric": "NDCG",
                "aggr_metric": "minmax",
                "value": ndcg_min_max
            },
            {
                "metric": "DCG",
                "aggr_metric": "mean",
                "value": np.mean(dcg_list)
            },
            {
                "metric": "DCG",
                "aggr_metric": "min",
                "value": np.amin(dcg_list)
            },
            {
                "metric": "DCG",
                "aggr_metric": "minmax",
                "value": dcg_min_max
            }
        ] if evaluation_ground_truth != "GROUP_CHOICES" else [
            {
                "metric" : "NDCG",
                "aggr_metric" : "mean",
                "value" : np.mean(ndcg_list)
            },
            {
                "metric" : "DCG",
                "aggr_metric" : "mean",
                "value" : np.mean(dcg_list)
            }
        ]


class BinaryEvaluator(MetricEvaluator):

    def evaluateUserBinary(self, user_ground_truth, group_recommendation, user_norm):
        correct_recs_list = user_ground_truth.loc[
            (user_ground_truth.index.isin(group_recommendation)) & (user_ground_truth.final_rating > 0)]
        correct_recs = correct_recs_list.shape[0]
        all_correct_per_user = user_ground_truth.loc[user_ground_truth.final_rating > 0].shape[0]
        if all_correct_per_user == 0:
            return (0.0, 0.0, 0.0, 0.0)
        recall = user_norm * correct_recs / all_correct_per_user

        # bounded recall, denominator is min(# of relevant items, length of the list)
        all_correct_per_user_caped = min([all_correct_per_user, len(group_recommendation)])
        bounded_recall = user_norm * correct_recs / all_correct_per_user_caped

        # discounted first hit
        if correct_recs == 0:
            dfh = 0.0
        else:
            for k, item in enumerate(group_recommendation):
                if item in correct_recs_list.index:
                    first_hit_rank = k
                    break
            dfh = user_norm * 1 / np.log2(first_hit_rank + 2)

        #mean reciprocal rank
        rank_positions = np.where(np.isin(group_recommendation, correct_recs_list.index.tolist()))
        reciprocal_ranks = [1/(x+1) for x in rank_positions]
        mean_reciprocal_rank = np.mean(reciprocal_ranks)
        
        return (recall, bounded_recall, dfh, mean_reciprocal_rank)

    def evaluateGroupRecommendation(
            self,
            group_ground_truth,
            group_recommendation,
            group_members,
            propensity_per_item,
            per_user_propensity_normalization_term,
            evaluation_ground_truth,            
            binarize_feedback_positive_threshold,
            binarize_feedback,
            feedback_polarity_debiasing
    ):
        # Irrespective of the binarize_feedback setting, we need binary feedback for this set of metrics
        # Use binary_positive_threshold, but do not use polarity_debiasing (kind of does the same thing)
        recall_list = list()
        bounded_recall_list = list()
        dfh_list = list()
        mrr_list = list()
        zero_recall = 0
        for user in group_members:
            user_ground_truth = group_ground_truth.loc[group_ground_truth['user'] == user]
            user_ground_truth.set_index("item", inplace=True)

            #feedback binarization
            if evaluation_ground_truth == "GROUP_CHOICES":
                user_ground_truth["final_rating"] = 1
            else:
                user_ground_truth["final_rating"] = 0
                user_ground_truth.loc[user_ground_truth.rating >= binarize_feedback_positive_threshold,"final_rating"] = 1

            # self-normalized inverse propensity debiasing
            user_ground_truth.loc[:, "final_rating"] = user_ground_truth.loc[:, "final_rating"] / propensity_per_item[
                "propensity_score"]
            user_norm = 1.0
            if per_user_propensity_normalization_term is not None:
                user_norm = per_user_propensity_normalization_term[user]
            #print(user_ground_truth.head(3))
            recall_user, bounded_recall_user, dfh_user, mrr_user = self.evaluateUserBinary(user_ground_truth,
                                                                                 group_recommendation, user_norm)
            if recall_user == 0:
                zero_recall += 1

            recall_list.append(recall_user)
            bounded_recall_list.append(bounded_recall_user)
            dfh_list.append(dfh_user)
            mrr_list.append(mrr_user)

        # failsafe for all negative results
        if np.amax(recall_list) > 0:
            rec_min_max = np.amin(recall_list) / np.amax(recall_list)
            bound_min_max = np.amin(bounded_recall_list) / np.amax(bounded_recall_list)
            dfh_min_max = np.amin(dfh_list) / np.amax(dfh_list)
            mrr_min_max = np.amin(mrr_list)/np.amax(mrr_list)
        else:
            rec_min_max = 0.0
            bound_min_max = 0.0
            dfh_min_max = 0.0
            mrr_min_max = 0.0

        return [
            {
                "metric": "Recall",
                "aggr_metric": "mean",
                "value": np.mean(recall_list)
            },
            {
                "metric": "Recall",
                "aggr_metric": "min",
                "value": np.amin(recall_list)
            },
            {
                "metric": "Recall",
                "aggr_metric": "minmax",
                "value": rec_min_max
            },
            {
                "metric": "BoundedRecall",
                "aggr_metric": "mean",
                "value": np.mean(bounded_recall_list)
            },
            {
                "metric": "BoundedRecall",
                "aggr_metric": "min",
                "value": np.amin(bounded_recall_list)
            },
            {
                "metric": "BoundedRecall",
                "aggr_metric": "minmax",
                "value": bound_min_max
            },
            {
                "metric": "DFH",
                "aggr_metric": "mean",
                "value": np.mean(dfh_list)
            },
            {
                "metric": "DFH",
                "aggr_metric": "min",
                "value": np.amin(dfh_list)
            },
            {
                "metric": "DFH",
                "aggr_metric": "minmax",
                "value": dfh_min_max
            },
            {
                "metric": "MRR",
                "aggr_metric": "mean",
                "value": np.mean(mrr_list)
            },
            {
                "metric": "MRR",
                "aggr_metric": "min",
                "value": np.amin(mrr_list)
            },
            {
                "metric": "MRR",
                "aggr_metric": "minmax",
                "value": mrr_min_max
            },            
            {
                "metric": "zRecall",
                "aggr_metric": "mean",
                "value": zero_recall / len(group_members)
            }
        ] if evaluation_ground_truth != "GROUP_CHOICES" else [
            {
                "metric" : "Recall",
                "aggr_metric" : "mean",
                "value" : np.mean(recall_list)
            },
            {
                "metric" : "BoundedRecall",
                "aggr_metric" : "mean",
                "value" : np.mean(bounded_recall_list)
            },
            {
                "metric" : "DFH",
                "aggr_metric" : "mean",
                "value" : np.mean(dfh_list)
            },
            {
                "metric" : "zRecall",
                "aggr_metric" : "mean",
                "value" : zero_recall / len(group_members)
            },
            {
                "metric" : "MRR",
                "aggr_metric" : "mean",
                "value" : np.mean(mrr_list)
            }
        ]



class BaselinesEvaluators(MetricEvaluator):
    def evaluateGroupRecommendation(
            self,
            group_ground_truth,
            group_recommendation,
            group_members,
            propensity_per_item,
            per_user_propensity_normalization_term,
            evaluation_ground_truth,            
            binarize_feedback_positive_threshold,
            binarize_feedback,
            feedback_polarity_debiasing
    ):
        return None
