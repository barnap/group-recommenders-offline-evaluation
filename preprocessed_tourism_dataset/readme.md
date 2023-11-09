# Tourism dataset

The datset contains the following files:

- ratings: contains individual user preferences in the form:
  - user, item, rating, rank 
  - ratings / ranks are in 1 to 10 range (the best rating is 10, the best rank clearly is 1)
  - both ratings and ranks are provided as in some data instances the rankings of options were collected, and in some ratings of options
  - ratings are easily transformed to partial rankings

- group_choices: contains group preferences or group ranks / choices in the form:
  - group_id, item, rank 
  - first and second preferred option for each group is provided

- group_composition: contains composition of each group in the form: 
  - group_id, list of members, group_size, similarity_score, group_similarity
  - each user can only be a member of one group
  - similarity_score normalised (range: 0-1) average of the spearman foot-rule* pairwise diversities (converted to similarities; code provided) 
  - group sizes: 2-5, relatively small groups
  - group similarity takes values "similar" or "divergent" (median split on the similarity score)

- user_features: contains various user features that were obtained prior to group discussions, in the form:
  - user_id, gender, age, opn, cns, ext, agr, est, experience
  - opn, cns, ext, agr, est scores for the Big Five Factors personality model
  - experience represents how familiar group members are with the choice set (destinations)

- user_feedback: contains user feedback about group choice, the process, and group 
  - user_id, choice_sat, diff, ident, sim
  - choice_set is group members' individual choice satisfaction, range: 1-5
  - diff is members' perceived difficulty of the decision-making proces, range: 1-5
  - ident is members' level of identification with the resto of the group, range: 1-5
  - sim is members' perceived similarity of preferences within the group, range: 1-5

## Using the dataset
In order to use this dataset please make sure to cite the following publications:

[1] Delic, A., Neidhardt, J., Nguyen, T.N. and Ricci, F., 2018. An observational user study for group recommender systems in the tourism domain. Information Technology & Tourism, 19(1), pp.87-116.

[2] Delic, A., Neidhardt, J., Nguyen, T.N., Ricci, F., Rook, L., Werthner, H. and Zanker, M., 2016, September. Observing group decision making processes. In Proceedings of the 10th ACM conference on recommender systems (pp. 147-150).
