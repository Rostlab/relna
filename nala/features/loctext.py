from nala.features.relations import EdgeFeatureGenerator
from nala.utils.graph import get_path, build_walks

class ProteinWordFeatureGenerator(EdgeFeatureGenerator):
    """
    Check for the presence of the word "protein" in the sentence. If the word
    "protein" is part of an entity, then it checks for dependencies from the
    head token of the entity to the word and vice versa.

    For the dependency path between the word "protein" and the head token, it
    also calculates the bag of words representation, masked text and parts of
    speech for each token in the path.

    :param feature_set: the feature set for the dataset
    :type feature_set: nala.structures.data.FeatureDictionary
    :param graphs: the graph representation for each sentence in the dataset
    :type graphs: dictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, graphs, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.graphs = graphs
        """a dictionary of graphs to avoid recomputation of path"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            protein_word_found = False
            for token in sentence:
                if token.is_entity_part(edge.part) and token.word.lower().find('protein') >= 0:
                    protein_word_found = True
                    token_from = token.features['dependency_from'][0]
                    if token_from == head1:
                        feature_name = '78_dependency_from_entity_to_protein_word_[0]'
                        self.add_to_feature_set(edge, feature_name)
                    for dependency_to in token.features['dependency_to']:
                        token_to = dependency_to[0]
                        if token_to == head1:
                            feature_name = '79_dependency_from_protein_word_to_entity_[0]'
                            self.add_to_feature_set(edge, feature_name)
                        path = get_path(token, head1, edge.part, edge.sentence_id, self.graphs)
                        if path == []:
                            path = [token, head1]
                        for tok in path:
                            feature_name = '80_PWPE_bow_masked_'+tok.masked_text(edge.part)+'_[0]'
                            self.add_to_feature_set(edge, feature_name)
                            feature_name = '81_PWPE_pos_'+tok.features['pos']+'_[0]'
                            self.add_to_feature_set(edge, feature_name)
                            feature_name = '82_PWPE_bow_'+tok.word+'_[0]'
                            self.add_to_feature_set(edge, feature_name)
                        all_walks = build_walks(path)
                        for dep_list in all_walks:
                            dep_path = ''
                            for dep in dep_list:
                                feature_name = '83_'+'PWPE_dep_'+dep[1]+'_[0]'
                                self.add_to_feature_set(edge, feature_name)
                                dep_path += dep[1]
                            feature_name = '84_PWPE_dep_full+'+dep_path+'_[0]'
                            self.add_to_feature_set(edge, feature_name)
                        for j in range(len(all_walks)):
                            dir_grams = ''
                            for i in range(len(path)-1):
                                cur_walk = all_walks[j]
                                if cur_walk[i][0] == path[i]:
                                    dir_grams += 'F'
                                else:
                                    dir_grams += 'R'
                            feature_name = '85_PWPE_dep_gram_'+dir_grams+'_[0]'
                            self.add_to_feature_set(edge, feature_name)
            if protein_word_found:
                feature_name = '86_protein_word_found_[0]'
                self.add_to_feature_set(edge, feature_name)
            else:
                feature_name = '87_protein_not_word_found_[0]'
                self.add_to_feature_set(edge, feature_name)


class LocationWordFeatureGenerator(EdgeFeatureGenerator):
    """
    Check each sentence for the presence of location words if the sentence
    contains an edge. These location words include ['location', 'localize'].

    :param feature_set: the feature set for the dataset
    :type feature_set: nala.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            location_word = False
            if edge.entity1.class_id == 'e_1':
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            for token in sentence:
                if not token.is_entity_part(edge.part) and \
                    ('location' in token.word.lower() or \
                    'localize' in token.word.lower()):
                    location_word = True
                    if head1.features['id']<token.features['id']<head2.features['id']:
                        feature_name = '88_localize_word_in_between_[0]'
                        self.add_to_feature_set(edge, feature_name)
            if (location_word):
                feature_name = '89_location_word_found_[0]'
                self.add_to_feature_set(edge, feature_name)
            else:
                feature_name = '90_location_word_not_found_[0]'
                self.add_to_feature_set(edge, feature_name)


class FoundInFeatureGenerator(EdgeFeatureGenerator):
    """
    Check for the presence of "found" and "in" in the sentence that contains
    the edge. The words must be present in that order and must be between the
    two entities.

    :param feature_set: the feature set for the dataset
    :type feature_set: nala.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            found_word = False
            in_word = False
            if edge.entity1.class_id == 'e_1':
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token
            for i in range(head1.features['id']+1, head2.features['id']):
                if edge.part.sentences[edge.sentence_id][i].word.lower() == 'found':
                    found_word = True
                if edge.part.sentences[edge.sentence_id][i].word.lower() == 'in':
                    in_word = True
            if found_word and in_word:
                feature_name = '91_found_in_[0]'
                self.add_to_feature_set(edge, feature_name)
