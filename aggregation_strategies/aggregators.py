import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class AggregationStrategy(ABC):

    @staticmethod
    def getAggregator(strategy):
        if strategy == "ADD":
            return AdditiveAggregator()
        elif strategy == "LMS":
            return LeastMiseryAggregator()
        elif strategy == "BASE":
            return BaselinesAggregator()
        elif strategy == "GFAR":
            return GFARAggregator()
        elif strategy == "EPFuzzDA":
            return EPFuzzDAAggregator()
        return None

    @abstractmethod
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        pass


class AdditiveAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        aggregated_df = group_ratings.groupby('item').sum()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['item'])
        return {"ADD": recommendation_list}


class LeastMiseryAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        # aggregate using least misery strategy
        aggregated_df = group_ratings.groupby('item').min()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['item'])
        return {"LMS": recommendation_list}


class BaselinesAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        # aggregate using least misery strategy
        aggregated_df = group_ratings.groupby('item').agg({"predicted_rating": [np.sum, np.prod, np.min, np.max]})
        aggregated_df = aggregated_df["predicted_rating"].reset_index()
        # additive

        add_df = aggregated_df.sort_values(by="sum", ascending=False).reset_index()[['item', 'sum']]
        add_recommendation_list = list(add_df.head(recommendations_number)['item'])
        # multiplicative
        mul_df = aggregated_df.sort_values(by="prod", ascending=False).reset_index()[['item', 'prod']]
        mul_recommendation_list = list(mul_df.head(recommendations_number)['item'])
        # least misery
        lms_df = aggregated_df.sort_values(by="amin", ascending=False).reset_index()[['item', 'amin']]
        lms_recommendation_list = list(lms_df.head(recommendations_number)['item'])
        # most pleasure
        mpl_df = aggregated_df.sort_values(by="amax", ascending=False).reset_index()[['item', 'amax']]
        mpl_recommendation_list = list(mpl_df.head(recommendations_number)['item'])
        return {
            "ADD": add_recommendation_list,
            "MUL": mul_recommendation_list,
            "LMS": lms_recommendation_list,
            "MPL": mpl_recommendation_list
        }


class GFARAggregator(AggregationStrategy):
    # implements GFAR aggregation algorithm. For more details visit https://dl.acm.org/doi/10.1145/3383313.3412232

    # create an index-wise top-k selection w.r.t. list of scores
    def select_top_n_idx(self, score_df, top_n, top='max', sort=True):
        if top != 'max' and top != 'min':
            raise ValueError('top must be either Max or Min')
        if top == 'max':
            score_df.loc[score_df.index, "predicted_rating_rev"] = -score_df["predicted_rating"]

        select_top_n = min(top_n, len(score_df)-1)
        top_n_ind = np.argpartition(score_df.predicted_rating_rev, select_top_n)[:select_top_n]
        top_n_df = score_df.iloc[top_n_ind]

        if sort:
            return top_n_df.sort_values("predicted_rating_rev")

        return top_n_df

    # borda count that is limited only to top-max_rel_items, if you are not in the top-max_rel_items, you get 0
    def get_borda_rel(self, candidate_group_items_df, max_rel_items):
        from scipy.stats import rankdata
        top_records = self.select_top_n_idx(candidate_group_items_df, max_rel_items, top='max', sort=False)

        rel_borda = rankdata(top_records["predicted_rating_rev"].values, method='max')
        # candidate_group_items_df.loc[top_records.index,"borda_score"] = rel_borda
        return (top_records.index, rel_borda)

    # runs GFAR algorithm for one group
    def gfar_algorithm(self, group_ratings, top_n: int, relevant_max_items: int, n_candidates: int):

        group_members = group_ratings.user.unique()
        group_size = len(group_members)

        localDF = group_ratings.copy()
        localDF["predicted_rating_rev"] = 0.0
        localDF["borda_score"] = 0.0
        localDF["p_relevant"] = 0.0
        localDF["prob_selected_not_relevant"] = 1.0
        localDF["marginal_gain"] = 0.0

        # filter-out completely irrelevant items to decrease computational complexity
        # top_candidates_ids_per_member = []
        # for uid in  group_members:
        #    per_user_ratings = group_ratings.loc[group_ratings.user == uid]
        #    top_candidates_ids_per_member.append(select_top_n_idx(per_user_ratings, n_candidates, sort=False)["item"].values)

        # top_candidates_idx = np.unique(np.array(top_candidates_ids_per_member))

        # get the candidate group items for each member
        # candidate_group_ratings = group_ratings.loc[group_ratings["items"].isin(top_candidates_idx)]

        for uid in group_members:
            per_user_candidates = localDF.loc[localDF.user == uid]
            borda_index, borda_score = self.get_borda_rel(per_user_candidates, relevant_max_items)
            localDF.loc[borda_index, "borda_score"] = borda_score

            total_relevance_for_users = localDF.loc[borda_index, "borda_score"].sum()
            localDF.loc[borda_index, "p_relevant"] = localDF.loc[borda_index, "borda_score"] / total_relevance_for_users

        selected_items = []

        # top-n times select one item to the final list
        for i in range(top_n):
            localDF.loc[:, "marginal_gain"] = localDF.p_relevant * localDF.prob_selected_not_relevant
            item_marginal_gain = localDF.groupby("item")["marginal_gain"].sum()
            # select the item with the highest marginal gain
            item_pos = item_marginal_gain.argmax()
            item_id = item_marginal_gain.index[item_pos]
            selected_items.append(item_id)

            # update the probability of selected items not being relevant
            for uid in group_members:
                winner_row = localDF.loc[((localDF["item"] == item_id) & (localDF["user"] == uid))]

                # only update if any record for user-item was found
                if winner_row.shape[0] > 0:
                    p_rel = winner_row["p_relevant"].values[0]
                    p_not_selected = winner_row["prob_selected_not_relevant"].values[0] * (1 - p_rel)

                    localDF.loc[localDF["user"] == uid, "prob_selected_not_relevant"] = p_not_selected

            # remove winning item from the list of candidates
            localDF.drop(localDF.loc[localDF["item"] == item_id].index, inplace=True)
        return selected_items

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        selected_items = self.gfar_algorithm(group_ratings, recommendations_number, 20, 500)
        return {"GFAR": selected_items}


class EPFuzzDAAggregator(AggregationStrategy):
    # implements EP-FuzzDA aggregation algorithm. For more details visit https://dl.acm.org/doi/10.1145/3450614.3461679

    def ep_fuzzdhondt_algorithm(self, group_ratings, top_n, member_weights=None):
        group_members = group_ratings.user.unique()
        all_items = group_ratings["item"].unique()
        group_size = len(group_members)

        if not member_weights:
            member_weights = [1. / group_size] * group_size
        member_weights = pd.DataFrame(pd.Series(member_weights, index=group_members))

        localDF = group_ratings.copy()

        candidate_utility = pd.pivot_table(localDF, values="predicted_rating", index="item", columns="user",
                                           fill_value=0.0)
        candidate_sum_utility = pd.DataFrame(candidate_utility.sum(axis="columns"))

        total_user_utility_awarded = pd.Series(np.zeros(group_size), index=group_members)
        total_utility_awarded = 0.

        selected_items = []
        # top-n times select one item to the final list
        for i in range(top_n):
            # print()
            # print('Selecting item {}'.format(i))
            # print('Total utility awarded: ', total_utility_awarded)
            # print('Total user utility awarded: ', total_user_utility_awarded)

            prospected_total_utility = candidate_sum_utility + total_utility_awarded  # pd.DataFrame items x 1

            # print(prospected_total_utility.shape, member_weights.T.shape)

            allowed_utility_for_users = pd.DataFrame(np.dot(prospected_total_utility.values, member_weights.T.values),
                                                     columns=member_weights.T.columns,
                                                     index=prospected_total_utility.index)

            # print(allowed_utility_for_users.shape)

            # cap the item's utility by the already assigned utility per user
            unfulfilled_utility_for_users = allowed_utility_for_users.subtract(total_user_utility_awarded,
                                                                               axis="columns")
            unfulfilled_utility_for_users[unfulfilled_utility_for_users < 0] = 0

            candidate_user_relevance = pd.concat([unfulfilled_utility_for_users, candidate_utility]).min(level=0)
            candidate_relevance = candidate_user_relevance.sum(axis="columns")

            # remove already selected items
            candidate_relevance = candidate_relevance.loc[~candidate_relevance.index.isin(selected_items)]
            item_pos = candidate_relevance.argmax()
            item_id = candidate_relevance.index[item_pos]

            # print(item_pos,item_id,candidate_relevance[item_id])

            # print(candidate_relevance.index.difference(candidate_utility.index))
            # print(item_id in candidate_relevance.index, item_id in candidate_utility.index)
            selected_items.append(item_id)

            winner_row = candidate_utility.loc[item_id, :]
            # print(winner_row)
            # print(winner_row.shape)
            # print(item_id,item_pos,candidate_relevance.max())
            # print(selected_items)
            # print(total_user_utility_awarded)
            # print(winner_row.iloc[0,:])

            total_user_utility_awarded.loc[:] = total_user_utility_awarded.loc[:] + winner_row

            total_utility_awarded += winner_row.values.sum()
            # print(total_user_utility_awarded)
            # print(total_utility_awarded)

        return selected_items

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        selected_items = self.ep_fuzzdhondt_algorithm(group_ratings, recommendations_number)
        return {"EPFuzzDA": selected_items}