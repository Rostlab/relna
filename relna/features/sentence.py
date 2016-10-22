from nalaf.features.relations import EdgeFeatureGenerator
from relna.features.relations import TokenFeatureGenerator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class NamedEntityCountFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Named Entity Count for each sentence that contains an edge

    :type entity_type: str
    :type mode: str
    :type feature_set: dict
    :type is_training_mode: bool
    """
    def __init__(self, entity_type):
        self.entity_type = entity_type
        """type of entity"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entities = edge.part.get_entities_in_sentence(edge.sentence_id, self.entity_type)
            feature_name = '1_' + self.entity_type + '_count_['+str(len(entities))+']'

            # TODO, USE add_to_feature_set
            if is_training_mode:
                if feature_name not in feature_set:
                    feature_set[feature_name] = len(feature_set.keys())+1
                edge.features[feature_set[feature_name]] = 1
            else:
                if feature_name in feature_set.keys():
                    edge.features[feature_set[feature_name]] = 1


class BagOfWordsFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Bag of Words representation for each sentence that contains an edge

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type stop_words: list[str]
    :type is_training_mode: bool
    """
    def __init__(self, stop_words=None):
        if stop_words is None:
            stop_words = stopwords.words('english')
        self.stop_words = stop_words
        """a list of stop words"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            bow_map = {}
            for token in sentence:
                if token.word not in self.stop_words and not token.features['is_punct']:
                    feature_name = '2_bow_text_' + token.word + '_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    if token.is_entity_part(edge.part):
                        bow_string = 'ne_bow_' + token.word + '_[0]'
                        if bow_string not in bow_map.keys():
                            bow_map[bow_string] = 0
                        bow_map[bow_string] = bow_map[bow_string]+1
            for key, value in bow_map.items():
                feature_name = '3_'+key
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value)


class StemmedBagOfWordsFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates stemmed Bag of Words representation for each sentence that contains
    an edge, using the function given in the argument.

    By default it uses Porter stemmer

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type stemmer: nltk.stem.PorterStemmer
    :type stop_words: list[str]
    :type is_training_mode: bool
    """

    def __init__(self, stop_words=[]):
        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""
        self.stop_words = stop_words
        """a list of stop words"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]

            if is_training_mode:
                for token in sentence:
                    if self.stemmer.stem(token.word) not in self.stop_words and not token.features['is_punct']:
                        feature_name = '4_bow_stem_' + self.stemmer.stem(token.word) + '_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


class SentenceFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate features for each sentence containing an edge
    """

    def __init__(self):
        self.token_feature_generator = TokenFeatureGenerator()
        """an instance of TokenFeatureGenerator"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            text_count = {}
            for token in sentence:
                ann_types = self.token_feature_generator.annotated_types(token, edge)
                for ann in ann_types:
                    if ann not in text_count.keys():
                        text_count[ann] = 0
                    text_count[ann] = text_count[ann]+1
            for key, value in text_count.items():
                feature_name = '5_'+key+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=value)


class WordFilterFeatureGenerator(EdgeFeatureGenerator):
    """
    Checks if the sentence containing an edge contains any of the words
    given in the list.

    Value of 1 means that the sentence contains that word
    Value of 0 means that the sentence does not contain the word

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type words: list[str]
    :type stem: bool
    :type is_training_mode: bool
    """
    def __init__(self, words, stem=True):
        self.words = words
        """a list of words to check for their presence in the sentence"""
        self.stem = stem
        """whether the words in the sentence and the list should be stemmed"""
        self.stemmer = PorterStemmer()


    def generate(self, dataset, feature_set, is_training_mode):
        if self.stem:
            stemmed_words = [self.stemmer.stem(word) for word in self.words]
            for edge in dataset.edges():
                sentence = edge.part.sentences[edge.sentence_id]
                for token in sentence:
                    if self.stemmer.stem(token.word) in stemmed_words:
                        feature_name = '6_word_filter_stem_' + self.stemmer.stem(token.word) + '_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

        else:
            for edge in dataset.edges():
                sentence = edge.part.sentences[edge.sentence_id]
                for token in sentence:
                    if token.word in self.words:
                        feature_name = '6_word_filter_' + token.word + '_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
