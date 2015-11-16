from nala.features.context import *
from nala.features.entityhead import *
from nala.features.loctext import *
from nala.features.path import *
from nala.features.sentence import *
from nala.features import FeatureGenerator
from nala.structures.data import FeatureDictionary
from nala.preprocessing.spliters import Splitter, NLTKSplitter
from nala.preprocessing.tokenizers import Tokenizer, NLTKTokenizer, TmVarTokenizer
from nala.preprocessing.parsers import Parser, BllipParser, SpacyParser
from nala.preprocessing.edges import SimpleEdgeGenerator
from spacy.en import English

class RelationExtractionPipeline:
    """
    Prepares an instance of a dataset by executing modules in fixed order.
        * Finally executes each feature generator in the order they were provided

    :param class1: the class of entity1
    :type class1: str
    :param class1: the class of entity2
    :type class1: str
    :param rel_type: the relation type between the two entities
    :type rel_type: str
    :param train: if the mode is training or testing
    :type train: bool
    :param feature_set: the feature_set of the original training data
    :type feature_set: str
    :param feature_generators: one or more modules responsible for generating features
    :type feature_generators: collections.Iterable[FeatureGenerator]
    """

    def __init__(self, class1, class2, rel_type, train=False, feature_set=None,
            splitter=None, tokenizer=None, parser=None, feature_generators=None):
        self.class1 = class1
        self.class2 = class2
        self.feature_set = feature_set
        self.train = train

        if self.train:
            self.feature_set = FeatureDictionary()
        else:
            if not feature_set or not isinstance(feature_set, FeatureDictionary):
                raise ValueError("Feature Set is None or not of type FeatureDictionary.")
            else:
                self.feature_set = feature_set

        if not splitter:
            splitter = NLTKSplitter()

        if isinstance(splitter, Splitter):
            self.splitter = splitter
        else:
            raise TypeError('not an instance that implements Splitter')

        if not tokenizer:
            tokenizer = TmVarTokenizer()

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('not an instance that implements Tokenizer')

        self.graphs = {}

        if feature_generators is None:
            # use default feature generators
            feature_generators = [NamedEntityCountFeatureGenerator(class1, self.feature_set, training_mode=self.train),
                                  NamedEntityCountFeatureGenerator(class2, self.feature_set, training_mode=self.train),
                                  BagOfWordsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  StemmedBagOfWordsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  SentenceFeatureGenerator(self.feature_set, training_mode=self.train),
                                  WordFilterFeatureGenerator(self.feature_set, ['interact', 'bind', 'colocalize'], training_mode=self.train),
                                  EntityHeadTokenFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenUpperCaseFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenDigitsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenLetterPrefixesFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenPunctuationFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenChainFeatureGenerator(self.feature_set, training_mode=self.train),
                                  LinearContextFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityOrderFeatureGenerator(self.feature_set, training_mode=self.train),
                                  LinearDistanceFeatureGenerator(self.feature_set, training_mode=self.train),
                                  IntermediateTokensFeatureGenerator(self.feature_set, training_mode=self.train),
                                  PathFeatureGenerator(self.feature_set, self.graphs, training_mode=self.train),
                                  ProteinWordFeatureGenerator(self.feature_set, self.graphs, training_mode=self.train),
                                  LocationWordFeatureGenerator(self.feature_set, training_mode=self.train),
                                  FoundInFeatureGenerator(self.feature_set, training_mode=self.train)
                                 ]
        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))
            self.feature_generators = feature_generators
        elif isinstance(feature_generators, FeatureGenerator):
            self.feature_generators = [feature_generators]
        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')

        if not parser:
            parser = BllipParser()
        if isinstance(parser, Parser):
            self.parser = parser
        else:
            raise TypeError('not an instance that implements Parser')

        self.edge_generator = SimpleEdgeGenerator(class1, class2, rel_type)

    def set_mode(self, mode=False, feature_generators=None):
        self.train = mode
        if feature_generators is None:
            feature_generators = [NamedEntityCountFeatureGenerator(self.class1, self.feature_set, training_mode=self.train),
                                  NamedEntityCountFeatureGenerator(self.class2, self.feature_set, training_mode=self.train),
                                  BagOfWordsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  StemmedBagOfWordsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  SentenceFeatureGenerator(self.feature_set, training_mode=self.train),
                                  WordFilterFeatureGenerator(self.feature_set, ['interact', 'bind', 'colocalize'], training_mode=self.train),
                                  EntityHeadTokenFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenUpperCaseFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenDigitsFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenLetterPrefixesFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenPunctuationFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityHeadTokenChainFeatureGenerator(self.feature_set, training_mode=self.train),
                                  LinearContextFeatureGenerator(self.feature_set, training_mode=self.train),
                                  EntityOrderFeatureGenerator(self.feature_set, training_mode=self.train),
                                  LinearDistanceFeatureGenerator(self.feature_set, training_mode=self.train),
                                  IntermediateTokensFeatureGenerator(self.feature_set, training_mode=self.train),
                                  PathFeatureGenerator(self.feature_set, self.graphs, training_mode=self.train),
                                  ProteinWordFeatureGenerator(self.feature_set, self.graphs, training_mode=self.train),
                                  LocationWordFeatureGenerator(self.feature_set, training_mode=self.train),
                                  FoundInFeatureGenerator(self.feature_set, training_mode=self.train)
                                 ]
        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))
                if not feature_generator.training_mode==mode:
                    raise ValueError('FeatureGenerator at index {} not set in the correct mode'.format(index))
            self.feature_generators = feature_generators
        elif isinstance(feature_generators, FeatureGenerator):
            if not feature_genenrators.training_mode==mode:
                raise ValueError('FeatureGenerator at index not set in the correct mode.')
            else:
                self.feature_generators = [feature_generators]
        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')

    def execute(self, dataset):
        self.splitter.split(dataset)
        self.tokenizer.tokenize(dataset)
        self.edge_generator.generate(dataset)
        self.parser.parse(dataset)
        dataset.label_edges()
        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset)
