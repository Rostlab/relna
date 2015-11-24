import operator
from math import log2

class FeatureSelection:
    """
    Base class for feature selection given the entire feature set

    :param feature_set: the feature set for the dataset
    :type feature_set: nala.structures.data.FeatureDictionary
    """
    def __init__(self, dataset, feature_set):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.dataset = dataset
        """the dataset"""

    def select(self, nbest=55, group=True, mode='avgIG'):
        features = {}
        feature_list = self.information_gain()
        counts, entropy = self.sum_across_generators(feature_list)
        if group and mode=='avgIG':
            averageIg = {}
            for key in entropy.keys():
                averageIg[key] = entropy[key] / counts[key]
            sorted_avg_Ig = sorted(averageIg.items(), key=operator.itemgetter(1))
            nbest_avg_Ig = []
            for i in range(nbest):
                item = sorted_avg_Ig[i]
                nbest_avg_Ig.append(str(item[0]))
            nbest_avg_Ig = tuple(nbest_avg_Ig)
            for key, value in self.feature_set.items():
                if key.startswith(nbest_avg_Ig):
                    features[key] = value
        return features

    def sum_across_generators(self, feature_list):
        counts = {}
        entropy = {}
        for item in feature_list:
            feature_name = item[0]
            generator = int(feature_name.split('_')[0])
            if generator not in counts.keys():
                counts[generator] = 0
            counts[generator]+=1
            if generator not in entropy.keys():
                entropy[generator] = 0
            entropy[generator]+=1
        return counts, entropy

    def information_gain(self):
        number_pos_instances = 0
        number_neg_instances = 0

        for edge in self.dataset.edges():
            if edge.target == 1:
                number_pos_instances += 1
            else:
                number_neg_instances += 1

        number_total_instances = number_pos_instances + number_neg_instances
        percentage_pos_instances = number_pos_instances / number_total_instances
        percentage_neg_instances = number_neg_instances / number_total_instances

        first_ent_component = -1 * (percentage_pos_instances * log2(percentage_pos_instances) + percentage_neg_instances * log2(percentage_neg_instances))
        feature_list = []
        for key, value in self.feature_set.items():
            feature_present_in_pos = 0
            feature_present_in_neg = 0
            feature_absent_in_pos = 0
            feature_absent_in_neg = 0
            total_feature_present = 0
            total_feature_absent = 0

            for edge in self.dataset.edges():
                if edge.target == 1:
                    if value in edge.features.keys():
                        feature_present_in_pos += 1
                        total_feature_present += 1
                    else:
                        feature_absent_in_pos += 1
                        total_feature_absent +=1
                if edge.target == -1:
                    if value in edge.features.keys():
                        feature_present_in_neg += 1
                        total_feature_present += 1
                    else:
                        feature_absent_in_neg += 1
                        total_feature_absent += 1

            percentage_pos_given_feature = 0
            percentage_neg_given_feature = 0
            if (total_feature_present > 0):
                percentage_pos_given_feature = feature_present_in_pos / total_feature_present
                percentage_neg_given_feature = feature_present_in_neg / total_feature_present

            percentage_pos_given_feature_log = 0
            percentage_neg_given_feature_log = 0
            if percentage_pos_given_feature > 0:
                percentage_pos_given_feature_log = log2(percentage_pos_given_feature)
            if percentage_neg_given_feature > 0:
                percentage_neg_given_feature_log = log2(percentage_neg_given_feature)

            second_emp_component_factor = percentage_pos_given_feature * percentage_pos_given_feature_log + \
                                percentage_neg_given_feature * percentage_neg_given_feature_log

            percentage_feature_given_pos = feature_present_in_pos / number_pos_instances
            percentage_feature_given_neg = feature_present_in_pos / number_neg_instances
            percentage_feature = percentage_feature_given_pos * percentage_pos_instances + \
                        percentage_feature_given_neg * percentage_neg_instances

            second_ent_component = percentage_feature * second_emp_component_factor
            percentage_pos_given_feature_component = 0
            percentage_neg_given_feature_component = 0
            if total_feature_absent>0:
                percentage_pos_given_feature_component = feature_absent_in_pos / total_feature_absent
                percentage_neg_given_feature_component = feature_absent_in_neg / total_feature_absent

            percentage_pos_given_feature_component_log = 0
            percentage_neg_given_feature_component_log = 0
            if percentage_pos_given_feature_component>0:
                percentage_pos_given_feature_component_log = log2(percentage_pos_given_feature_component)
            if percentage_neg_given_feature_component>0:
                percentage_neg_given_feature_component_log = log2(percentage_neg_given_feature_component)

            third_component_multi_factor = percentage_pos_given_feature_component * percentage_pos_given_feature_component_log + \
                    percentage_neg_given_feature_component * percentage_neg_given_feature_component_log

            percentage_feature_comp_given_pos = feature_absent_in_pos / number_pos_instances
            percentage_feature_comp_given_neg = feature_absent_in_neg / number_neg_instances
            percentage_feature_comp = percentage_feature_comp_given_pos * percentage_pos_instances + \
                        percentage_feature_comp_given_neg * percentage_neg_instances

            third_ent_component = percentage_feature_comp * third_component_multi_factor
            entropy = first_ent_component + second_ent_component + third_ent_component

            feature_list.append([key, value, entropy])
        return feature_list
