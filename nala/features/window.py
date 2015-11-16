from nala.features import FeatureGenerator


class WindowFeatureGenerator(FeatureGenerator):
    """
    Generates window CRF features based on the value of a feature for the current token.

    For each token t and each feature 'NAME[0]',
    where NAME is the name of the feature and [0] indicates that it corresponds to the current token
    we generate by default at most 4 more additional features, corresponding to:
        * NAME[1], NAME[2] = the values of the features for the next two tokens in the sequence
        * NAME[-1], NAME[-2] = the values of the features for the previous two tokens in the sequence

    Which previous/next tokens are taken into consideration
    can be controlled with passing a template at the constructor.

    Expects some features to be already generated by other FeatureGenerators with keys 'NAME[0]'
    Implements the abstract class FeatureGenerator.
    """

    def __init__(self, template=(-2, -1, 1, 2), include_list=None):
        self.template = template
        """
        Controls which previous/next tokens are taken into consideration.
        The default value is (-2, -1, 1, 2) which mean we take into consideration
        the previous two and the next two tokens relative to the current one.
        """
        self.include_list = include_list
        """
        Controls which features from the previous/next tokens are taken into consideration.
        The default value is None which means we take into consideration all features per token.
        If you want to consider the value of only specific features, provide the names of those
        features in this list.
        """

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for sentence in dataset.sentences():
            for index, token in enumerate(sentence):

                if self.include_list is None:
                    feature_names = list(token.features.keys())
                else:
                    feature_names = [name for name in token.features.keys() if name in self.include_list]

                for feature_name in feature_names:
                    if feature_name.endswith('[0]'):
                        for template_index in self.template:
                            if -1 < index + template_index < len(sentence):
                                token.features['{}[{}]'.format(feature_name[:-3], template_index)] = \
                                    sentence[index + template_index].features[feature_name]