from nalaf.features.relations import EdgeFeatureGenerator
from nltk.stem import PorterStemmer

class LinearContextFeatureGenerator(EdgeFeatureGenerator):
    """
    Each entity has a precalculated entity head token. If the entity has only
    one token, that forms the head token. However, if the entity has multiple
    tokens, a simple heuristic based method is employed to select the token
    closest to the root of the sentence as the head token.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param linear_context: the number of words after and before the current
            token to consider for linear context
    :type linear_context: int
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, linear_context=3, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.linear_context = linear_context
        """the window size for the linear context"""
        self.training_mode = training_mode
        """whether the mode is training of testing"""

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            for i in range(1, self.linear_context+1):
                if head1.features['id'] < len(sentence):
                    if (head1.features['id']+i)<len(sentence):
                        self.linear_order_features('entity1_linear_'+str(i)+'_',
                                sentence[head1.features['id']+i], edge, sentence)
                if head2.features['id'] < len(sentence):
                    if (head2.features['id']+i)<len(sentence):
                        self.linear_order_features('entity2_linear_'+str(i)+'_',
                                sentence[head2.features['id']+i], edge, sentence)
                if head1.features['id'] >= 0:
                    if (head1.features['id']-i)>=0:
                        self.linear_order_features('entity1_linear_-'+str(i)+'_',
                                sentence[head1.features['id']-i], edge, sentence)
                if head2.features['id'] >= 0:
                    if (head2.features['id']-i)>=0:
                        self.linear_order_features('entity2_linear_-'+str(i)+'_',
                                sentence[head2.features['id']-i], edge, sentence)

    def linear_order_features(self, prefix, token, edge, sentence):
        feature_name_1 = '23_' + prefix + 'txt_' + token.word + '_[0]'
        feature_name_2 = '24_' + prefix + 'pos_' + token.features['pos'] + '_[0]'
        feature_name_3 = '25_' + prefix + 'given_[0]'
        feature_name_4 = '26_' + prefix + 'txt_' + token.masked_text(edge.part) + '_[0]'
        feature_name_5 = '27_' + prefix + 'ann_type_entity_[0]'

        self.add_to_feature_set(edge, feature_name_1)
        self.add_to_feature_set(edge, feature_name_2)
        if token.is_entity_part(edge.part):
            self.add_to_feature_set(edge, feature_name_3)
        self.add_to_feature_set(edge, feature_name_4)
        if token.is_entity_part(edge.part):
            entity = token.get_entity(edge.part)
            feature_name_6 = '28_' + prefix + 'ann_type_' + entity.class_id + '_[0]'
            self.add_to_feature_set(edge, feature_name_5)
            self.add_to_feature_set(edge, feature_name_6)


class EntityOrderFeatureGenerator(EdgeFeatureGenerator):
    """
    The is the order of the entities in the sentence.  Whether entity1 occurs
    first or entity2 occurs first.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            if edge.entity1.offset < edge.entity2.offset:
                feature_name = '30_order_entity1_entity2_[0]'
            else:
                feature_name = '30_order_entity2_entity1_[0]'
            self.add_to_feature_set(edge, feature_name)


class LinearDistanceFeatureGenerator(EdgeFeatureGenerator):
    """
    The absolute distance between the two entities in the edge.
    If distance is greater than 5, add to feature set.
    Also add the actual distance between the two entities.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param distance: the number of tokens between the two entities, default 5
    :type distance: int
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, distance=5, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.distance = distance
        """the distance parameter"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entity1_number = edge.entity1.head_token.features['id']
            entity2_number = edge.entity2.head_token.features['id']
            distance = abs(entity1_number-entity2_number)
            if distance>self.distance:
                feature_name = '31_entity_linear_distance_greater_than_[5]'
                self.add_to_feature_set(edge, feature_name)
            else:
                feature_name = '31_entity_linear_distance_lesser_than_[5]'
                self.add_to_feature_set(edge, feature_name)
            feature_name = '32_entity_linear_distance_[0]'
            self.add_to_feature_set(edge, feature_name, value=distance)


class IntermediateTokensFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate the bag of words representation, masked text, stemmed text and
    parts of speech tag for each of the tokens present between two entities in
    an edge.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""
        self.stemmer = PorterStemmer()
        """an instance of PorterStemmer"""

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            if edge.entity1.head_token.features['id'] < edge.entity2.head_token.features['id']:
                first = edge.entity1.head_token.features['id']
                second = edge.entity2.head_token.features['id']
                for i in range(first+1, second):
                    token = sentence[i]
                    feature_name = '33_fwd_bow_intermediate_'+token.word+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '34_fwd_bow_intermediate_masked_'+token.masked_text(edge.part)+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '35_fwd_stem_intermediate_'+self.stemmer.stem(token.word)+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '36_fwd_pos_intermediate_'+token.features['pos']+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
            else:
                first = edge.entity2.head_token.features['id']
                second = edge.entity1.head_token.features['id']
                for i in range(first+1, second):
                    token = sentence[i]
                    feature_name = '37_bkd_bow_intermediate_'+token.word+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '38_bkd_bow_intermediate_masked_'+token.masked_text(edge.part)+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '39_bkd_stem_intermediate_'+self.stemmer.stem(token.word)+'_[0]'
                    self.add_to_feature_set(edge, feature_name)
                    feature_name = '40_bkd_pos_intermediate_'+token.features['pos']+'_[0]'
                    self.add_to_feature_set(edge, feature_name)

            for i in range(first+1, second):
                token = sentence[i]
                feature_name = '41_bow_intermediate_'+token.word+'_[0]'
                self.add_to_feature_set(edge, feature_name)
                feature_name = '42_bow_intermediate_masked_'+token.masked_text(edge.part)+'_[0]'
                self.add_to_feature_set(edge, feature_name)
                feature_name = '43_stem_intermediate_'+self.stemmer.stem(token.word)+'_[0]'
                self.add_to_feature_set(edge, feature_name)
                feature_name = '44_pos_intermediate_'+token.features['pos']+'_[0]'
                self.add_to_feature_set(edge, feature_name)
