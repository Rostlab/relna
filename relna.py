import argparse
import os
import cPickle as pickle

from spacy.en import English
from nala.preprocessing.parsers import SpacyParser
from nala.utils.readers import HTMLReader
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.structures.relation_pipelines import RelationExtractionPipeline
from nala.learning.svmlight import SVMLightTreeKernels
from nala.learning.taggers import LocTextRelationExtractor
from nala.learning.evaluators import DocumentLevelRelationEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple demo for using the relna pipeline for prediction')

    parser.add_argument('-s', '--svmlight_dir', help='path to the directory containing the svmlight executables',
                        required=True)

    parser.add_argument('--color', help='uses color for highlighting predictions if supported '
                                        'otherwise prints them in new line',
                        action='store_true', default=True, dest='color')
    parser.add_argument('--no_color', help='prints predictions in new line',
                        action='store_false', dest='color')

    parser.add_argument('-o', '--output_dir', help='write the output to the provided directory, '
                                                   'the format can be specified with the -f switch, '
                                                   'otherwise the output will be written to the standard console')
    parser.add_argument('-f', '--file_format', help='the format for writing the output to a directory',
                        choices=['ann.json', 'pubtator'], default='ann.json')
    parser.add_argument('-p', '--feature_set', help='the feature set of the training set stored as a pickle file.')
    parser.add_argument('-d', '--dir', help='directory or file you want to predict for', required=True)
    parser.add_argument('-e', '--entities', nargs='+' help='classes for the two entities', required=True)
    parser.add_argument('-r', '--relation', help='type of relation between the two entities', required=True)
    args = parser.parse_args()

    warning = 'Due to a dependence on GNormPlus, running nala with -s and -d switches might take a long time.'
    try:
        dataset = HTMLReader(args.dir).read()
        AnnJsonAnnotationReader(args.dir, read_just_mutations=False, read_relations=True).annotate(dataset)
    else:
        raise FileNotFoundError('directory or file "{}" does not exist'.format(args.dir))

    if os.path.isfile(args.feature_set):
        with open(args.feature_set, 'rb') as fp:
            feature_set = pickle.load(fp)
    else IOError:
        raise FileNotFoundError('pickle file {} for feature set not found'.format(args.feature_set))

    nlp = English()

    pipeline = RelationExtractionPipeline(args.entities[0], args.entities[1], args.relation, parser=SpacyParser(nlp))
    pipeline.execute()

    # get the predictions
    svmlight = SVMLightTreeKernels(args.svmlight_dir, use_tree_kernel=False)
    tagger = LocTextRelationExtractor(args.entities[0], args.entities[1], args.relation, svmlight)
    tagger.tag(test, feature_set)

    StubSameSentenceRelationExtractor().tag(dataset)

    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            raise NotADirectoryError('{} is not a directory'.format(args.output_dir))
        if args.file_format == 'ann.json':
            TagTogFormat(dataset, to_save_to=args.output_dir).export(threshold_val=0)
        elif args.file_format == 'pubtator':
            PubTatorFormat(dataset, location=os.path.join(args.output_dir, 'pubtator.txt')).export()
    else:
        ConsoleWriter(args.color).write(dataset)
