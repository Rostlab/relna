import argparse
import os
import pickle

import pkg_resources

from relna.utils import PRO_CLASS_ID, MUT_CLASS_ID, PRO_REL_MUT_CLASS_ID
from nalaf.utils.readers import TextFilesReader, PMIDReader
from nalaf.utils.readers import StringReader
from nalaf.utils.writers import TagTogFormat, PubTatorFormat
from relna.utils.writers import RelnaConsoleWriter
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.learning.svmlight import SVMLightTreeKernels
from relna.learning.taggers import TranscriptionFactorTagger
from relna.learning.taggers import RelnaRelationExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple demo for using the relna pipeline for prediction')

    parser.add_argument('-c', '--svmlight_dir', help='path to the directory containing the svmlight executables',
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

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-s', '--string', help='string you want to predict for')
    group.add_argument('-d', '--dir_or_file', help='directory or file you want to predict for')
    group.add_argument('-p', '--pmids', nargs='+', help='a single PMID or a list of PMIDs separated by space')

    args = parser.parse_args()

    warning = 'Due to a dependence on GNormPlus, running nala with -s and -d switches might take a long time.'
    if args.string:
        print(warning)
        dataset = StringReader(args.string).read()
    elif args.pmids:
        dataset = PMIDReader(args.pmids).read()
    elif os.path.exists(args.dir_or_file):
        print(warning)
        dataset = TextFilesReader(args.dir_or_file).read()
    else:
        raise FileNotFoundError('directory or file "{}" does not exist'.format(args.dir_or_file))

    TranscriptionFactorTagger(pkg_resources.resource_filename('relna.data', 'goose.dat')).tag(dataset, uniprot=True)

    with open(pkg_resources.resource_filename('relna.data', 'features.pickle'), 'rb') as fp:
        feature_set = pickle.load(fp)

    pipeline = RelationExtractionPipeline(PRO_CLASS_ID, MUT_CLASS_ID, PRO_REL_MUT_CLASS_ID, feature_set=feature_set)
    pipeline.execute(dataset, feature_generators=RelnaRelationExtractor.default_feature_generators(PRO_CLASS_ID, MUT_CLASS_ID, feature_set, train=False))

    # get the predictions
    svmlight = SVMLightTreeKernels(args.svmlight_dir, use_tree_kernel=False)
    tagger = RelnaRelationExtractor(PRO_CLASS_ID, MUT_CLASS_ID, PRO_REL_MUT_CLASS_ID, svmlight)
    tagger.tag(dataset, feature_set)

    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            raise NotADirectoryError('{} is not a directory'.format(args.output_dir))
        if args.file_format == 'ann.json':
            TagTogFormat(dataset, to_save_to=args.output_dir).export(threshold_val=0)
        elif args.file_format == 'pubtator':
            PubTatorFormat(dataset, location=os.path.join(args.output_dir, 'pubtator.txt')).export()
    else:
        RelnaConsoleWriter(args.color).write(dataset)
