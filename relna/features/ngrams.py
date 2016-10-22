from nalaf.features.relations import EdgeFeatureGenerator

class BiGramFeatureGenerator(EdgeFeatureGenerator):
    """
    For each edge, we consider all the intermediate tokens between the two
    entities. For all the tokens between the entities, we construct an n-gram
    representation.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self):
        pass


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            if edge.entity1.offset < edge.entity2.offset:
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token
            for i in range(head1.features['id'], head2.features['id']):
                token1 = edge.part.sentences[edge.sentence_id][i]
                token2 = edge.part.sentences[edge.sentence_id][i+1]
                feature_name = '92_bigram_'+token1.word+'_'+token2.word+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = '93_bigram_pos_'+token1.features['pos']+'_'+token2.features['pos']+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = '93_bigram_pos_'+token1.features['dep']+'_'+token2.features['dep']+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


class TriGramFeatureGenerator(EdgeFeatureGenerator):
    """
    For each edge, we consider all the intermediate tokens between the two
    entities. For all the tokens between the entities, we construct an n-gram
    representation.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self):
        pass


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            if edge.entity1.offset < edge.entity2.offset:
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token
            if head2.features['id']-head1.features['id']==2:
                for i in range(head1.features['id'], head2.features['id']):
                    token1 = edge.part.sentences[edge.sentence_id][i]
                    token2 = edge.part.sentences[edge.sentence_id][i+1]
                    token3 = edge.part.sentences[edge.sentence_id][i+2]
                    feature_name = '92_trigram_'+token1.word+'_'+token2.word+'_'+token3.word+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = '93_trigram_pos_'+token1.features['pos']+'_'+token2.features['pos']+'_'+token3.features['pos']+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = '93_trigram_pos_'+token1.features['dep']+'_'+token2.features['dep']+'_'+token3.features['dep']+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
