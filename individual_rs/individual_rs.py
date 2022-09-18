import numpy as np
import pandas as pd

from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from sklearn.model_selection import train_test_split
from lenskit import batch, topn, util
import settings.config as cfg

from abc import ABC, abstractmethod


class IndividualRS(ABC):

    @staticmethod
    def train_individual_rs_and_get_predictions(training_df, test_df):
        if cfg.individual_rs_strategy == "LENSKIT_CF_USER":
            print(cfg.individual_rs_strategy)
            rs =  ItemItemKNN()
            return rs.train_and_predict(training_df, test_df)
        if cfg.individual_rs_strategy == "LENSKIT_CF_ITEM":
            print(cfg.individual_rs_strategy)
            rs =  UserUserKNN()
            return rs.train_and_predict(training_df, test_df)  
        return None    
    
    @staticmethod    
    def evalAlg(aname, params, algo, train, test):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)
        users = test.user.unique()
        # now we run the recommender
        recs = batch.recommend(fittable, users, cfg.recommendations_number)
        # add the algorithm name for analyzability
        recs['Algorithm'] = aname
        recs['Params'] = params
        return recs
    
    @staticmethod
    def hyperparam_eval(algName, training_df, paramList):    
        all_recs = []
        hp_training_df, validation_df = train_test_split(training_df, test_size=0.2, random_state=42, shuffle=True, stratify=training_df["user"])
        for param in paramList:
            if algName == "LENSKIT_CF_USER":
                alg = UserUser(param)
            elif algName == "LENSKIT_CF_ITEM":
                alg = ItemItem(param)
                
            all_recs.append(IndividualRS.evalAlg(algName, param, alg, hp_training_df, validation_df))
        all_recs = pd.concat(all_recs, ignore_index=True)
        
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        results = rla.compute(all_recs, validation_df)
        mean_res = results.reset_index().groupby(["Algorithm","Params"]).mean()["ndcg"]
        maxid = mean_res.argmax()
        return mean_res.index[maxid]


    @abstractmethod
    def train_and_predict(self, training_df, test_df):
        pass

class UserUserKNN(IndividualRS):
    # Train lenskit CF user-user individual recommender system and predict ratings
    def train_and_predict(self, training_df, test_df):
        if cfg.individual_rs_validation_folds_k <=0:
            print("training")        
            best_hyperparam = IndividualRS.hyperparam_eval(cfg.individual_rs_strategy, training_df, [1,5,10,20,30,40,50])
            nNeighbors = best_hyperparam[1]
            print("nNeighbors hyperparameter:"+str(nNeighbors))
            
            user_user = UserUser(nNeighbors)  
            recsys = Recommender.adapt(user_user)
            recsys.fit(training_df)
            
            print("evaluating predictions")
            # Evaluating predictions 
            test_df['predicted_rating'] = recsys.predict(test_df)
            print("Done!")
            return test_df
        return None 
        
class ItemItemKNN(IndividualRS):
    # Train lenskit CF user-user individual recommender system and predict ratings
    def train_and_predict(self, training_df, test_df):
        if individual_rs_validation_folds_k <=0:
            print("training")
            best_hyperparam = hyperparam_eval(cfg.individual_rs_strategy, training_df, [1,5,10,20,30,40,50])
            nNeighbors = best_hyperparam[1]
            print("nNeighbors hyperparameter:"+str(nNeighbors))
            
            item_item = ItemItem(nNeighbors) 
            recsys = Recommender.adapt(item_item)
            recsys.fit(training_df)
            
            print("evaluating predictions")
            # Evaluating predictions 
            test_df['predicted_rating'] = recsys.predict(test_df)
            print("Done!")
            return test_df
        return None           