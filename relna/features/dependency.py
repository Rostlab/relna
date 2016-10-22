from nalaf.features.relations import EdgeFeatureGenerator

class DependencyChainFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate the dependency chain for each token in the sentence containing an
    edge

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type nlp: spacy.en.English
    :type training_mode: bool
    """
    def __init__(self, feature_set, nlp, training_mode=True):
        self.feature_set = feature_set
        """Feature set for the dataset"""
        self.nlp = nlp
        """an instance of spacy.en.English"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sent = edge.part.get_sentence_string_array()[edge.sentence_id]
            parsed = self.nlp(sent)
            sentence = next(sent.sents)
            dependencies = {}
            for token in sentence:
                token_dependency = self.dependency_labels_to_root(token)
                feature_name_1 = '22_dep_chain_len_[' + len(token_dependency) +']'
                if self.training_mode:
                    if feature_name_1 not in self.feature_set.keys():
                        self.feature_set[feature_name_1] = len(self.feature_set.keys())
                        edge.features[self.feature_set[feature_name_1]] = 0
                    edge.features[self.feature_set[feature_name_1]] += 1
                else:
                    if feature_name_1 in self.feature_set.keys():
                        if self.feature_set[feature_name_1] not in edge.features.keys():
                            edge.features[self.feature_set[feature_name_1]] = 0
                        edge.features[self.feature_set[feature_name_1]] += 1

    def dependency_labels_to_root(self, token):
        """Walk up the syntactic tree, collecting the arc labels."""
        dep_labels = []
        token = token.head
        while token.head is not token:
            dep_labels.append((token.orth_, token.dep_))
            token = token.head
        return dep_labels
