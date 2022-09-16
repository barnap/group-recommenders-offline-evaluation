from abc import ABC, abstractmethod

import random
import numpy as np
import settings.config as cfg

random.seed(1010101)

# generation of random, similar and divergent groups revised from Kaya et al
# https://github.com/mesutkaya/recsys2020/blob/8a8c7088bebc3309b8517f62248386ea7be39776/GFAR_python/create_group.py

class GroupsGenerator(ABC):

    @staticmethod
    def getGroupsGenerator(type):
        if type == "RANDOM":
            return RandomGroupsGenerator()
        elif type == "SIMILAR":
            return SimilarGroupsGenerator()
        elif type == "DIVERGENT":
            return DivergentGroupsGenerator()
        elif type == "SIMILAR_ONE_DIVERGENT":
            return MinorityGroupsGenerator()
        return None

    @staticmethod
    def compute_average_similarity(group, user_id_indexes, sim_matrix):
        similarities = list()
        for user_1 in group:
            user_1_index = user_id_indexes.tolist().index(user_1)
            for user_2 in group:
                user_2_index = user_id_indexes.tolist().index(user_2)
                if user_1 != user_2:
                    similarities.append(sim_matrix[user_1_index][user_2_index])
        return np.mean(similarities)

    @abstractmethod
    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        pass


class RandomGroupsGenerator(GroupsGenerator):

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            for i in range(group_number_to_create):
                group = random.sample(user_id_set, group_size)
                groups_list.append(
                    {
                        "group_size": group_size,
                        "group_similarity": 'random',
                        "group_members": group,
                        "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                    }
                )
            print(len(groups_list))
        return groups_list


class SimilarGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_user_for_sim_group(group, sim_matrix, user_id_indexes, sim_threshold=0.4):
        '''
        Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
        selects from the remaining users that has a PCC value >= sim_threshold to any of the existing members.
        :param group:
        :param sim_matrix:
        :param user_id_indexes:
        :param sim_threshold:
        :return:
        '''
        ids_to_select_from = set()
        for member in group:
            member_index = user_id_indexes.tolist().index(member)
            indexes = np.where(sim_matrix[member_index] >= sim_threshold)[0].tolist()
            user_ids = [user_id_indexes[index] for index in indexes]
            ids_to_select_from = ids_to_select_from.union(set(user_ids))
        candidate_ids = ids_to_select_from.difference(set(group))
        if len(candidate_ids) == 0:
            return None
        else:
            selection = random.sample(candidate_ids, 1)
            return selection[0]

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < group_size:
                    new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes,
                                                                                  sim_threshold=cfg.similar_threshold)
                    if new_member is None:
                        break
                    group.append(new_member)
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'similar',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class DivergentGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_user_for_divergent_group(group, sim_matrix, user_id_indexes, sim_threshold=0.0):
        '''
        Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
        selects from the remaining users that has a PCC value < sim_threshold to any of the existing members.
        :param group:
        :param sim_matrix:
        :param user_id_indexes:
        :param sim_threshold:
        :return:
        '''
        ids_to_select_from = set()
        for member in group:
            member_index = user_id_indexes.tolist().index(member)
            indexes = np.where(sim_matrix[member_index] < sim_threshold)[0].tolist()
            user_ids = [user_id_indexes[index] for index in indexes]
            ids_to_select_from = ids_to_select_from.union(set(user_ids))
        candidate_ids = ids_to_select_from.difference(set(group))
        if len(candidate_ids) == 0:
            return None
        else:
            selection = random.sample(candidate_ids, 1)
            return selection[0]

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < group_size:
                    new_member = DivergentGroupsGenerator.select_user_for_divergent_group(group, similarity_matrix,
                                                                                     user_id_indexes,
                                                                                     sim_threshold=cfg.dissimilar_threshold)
                    if new_member is None:
                        break
                    group.append(new_member)
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'divergent',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class MinorityGroupsGenerator(GroupsGenerator):
    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < (group_size - 1):
                    new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix,
                                                                                     user_id_indexes,
                                                                                     sim_threshold=cfg.similar_threshold)
                    if new_member is None:
                        break
                    group.append(new_member)

                dissimilar_member = DivergentGroupsGenerator.select_user_for_divergent_group(group, similarity_matrix,
                                                                                              user_id_indexes,
                                                                                              sim_threshold=cfg.dissimilar_threshold)
                if dissimilar_member is not None:
                    group.append(dissimilar_member)
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'similar_one_divergent',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list

