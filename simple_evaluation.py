from nalaf.utils.readers import HTMLReader
from nalaf.utils.annotation_readers import AnnJsonAnnotationReader
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.learning.svmlight import SVMLightTreeKernels
from nalaf.preprocessing.parsers import SpacyParser
from spacy.en import English
# from relna.learning.taggers import TranscriptionFactorTagger
from relna.learning.taggers import RelnaRelationExtractor
import argparse

parser = argparse.ArgumentParser(description='Simple-evaluate relna corpus corpus')

parser.add_argument('--corpus', required=True, choices=["relna"])
parser.add_argument('--use_tk', default=False, action='store_true')
parser.add_argument('--use_test_set', default=False, action='store_true')

args = parser.parse_args()

print(args)

# ------------------------------------------------------------------------------

k = 5

if args.use_tk:
    svm_folder = ''  # '/usr/local/manual/svm-light-TK-1.2.1/' -- must be in your path
    nlp = English(entity=False)
    parser = SpacyParser(nlp, constituency_parser=True)
else:
    svm_folder = ''  # '/usr/local/manual/bin/' -- must be in your path
    parser = None

if args.corpus == "relna":
    # Relna
    dataset_folder_html = '/Users/jmcejuela/Work/hck/relna/resources/corpora/relna/corrected/'
    dataset_folder_annjson = dataset_folder_html
    rel_type = 'r_4'


def read_dataset():
    dataset = HTMLReader(dataset_folder_html).read()
    AnnJsonAnnotationReader(
            dataset_folder_annjson,
            read_relations=True,
            read_only_class_id=None,
            delete_incomplete_docs=False).annotate(dataset)

    return dataset

dataset = read_dataset()
tagger = StubSameSentenceRelationExtractor('e_1', 'e_2', rel_type)
evaluator = DocumentLevelRelationEvaluator(rel_type=rel_type, match_case=False)


print("# FOLDS")
merged = []
for fold in range(k):
    training, validation, test = dataset.cv_kfold_split(k, fold, validation_set=(not args.use_test_set))
    if args.use_test_set:
        validation = test

    tagger.tag(validation)

    r = evaluator.evaluate(validation)
    merged.append(r)
    print(r)

print("\n# FINAL")
ret = Evaluations.merge(merged)
print(ret)


print("\n\n\n\n\n")


dataset = read_dataset()
pipeline = RelationExtractionPipeline('e_1', 'e_2', rel_type, parser=parser)
evaluator = DocumentLevelRelationEvaluator(rel_type=rel_type, match_case=False)


print("# FOLDS")
merged = []
for fold in range(k):

    training, validation, test = dataset.cv_kfold_split(k, fold, validation_set=(not args.use_test_set))
    if args.use_test_set:
        validation = test

    # Learn
    pipeline.execute(training, train=True, feature_generators=RelnaRelationExtractor.default_feature_generators('e_1', 'e_2', pipeline.feature_set, train=True))
    svmlight = SVMLightTreeKernels(svmlight_dir_path=svm_folder, use_tree_kernel=args.use_tk)
    instancesfile = svmlight.create_input_file(training, 'train', pipeline.feature_set)
    svmlight.learn(instancesfile)

    # Predict & Read predictions
    pipeline.execute(validation, train=False, feature_generators=RelnaRelationExtractor.default_feature_generators('e_1', 'e_2', pipeline.feature_set, train=False))
    instancesfile = svmlight.create_input_file(validation, 'test', pipeline.feature_set)
    predictionsfile = svmlight.tag(instancesfile)
    svmlight.read_predictions(validation, predictionsfile)

    results = evaluator.evaluate(validation)
    merged.append(results)
    print(results)

print("\n# FINAL")
ret = Evaluations.merge(merged)
print(ret)


# Expected output, something like:

# python simple_evaluation.py --corpus relna
# Namespace(corpus='relna', use_test_set=False, use_tk=False)
# # FOLDS
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	51	261	0	0	0	0.1635	1.0000	0.2810	0.0034	0.1635	1.0000	0.2810	0.0034
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	48	187	1	0	0	0.2043	0.9796	0.3380	0.0048	0.2043	0.9796	0.3380	0.0050
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	49	136	0	0	0	0.2649	1.0000	0.4188	0.0049	0.2649	1.0000	0.4188	0.0048
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	58	187	0	0	0	0.2367	1.0000	0.3828	0.0037	0.2367	1.0000	0.3828	0.0039
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	49	196	0	0	0	0.2000	1.0000	0.3333	0.0044	0.2000	1.0000	0.3333	0.0046
#
# # FINAL
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	255	967	1	0	0	0.2087	0.9961	0.3451	0.0018	0.2087	0.9961	0.3451	0.0017
#
#
#
#
#
#
# # FOLDS
# Processing [SpaCy] |################################| 252/252
# Processing [SpaCy] |################################| 84/84
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	39	45	12	0	0	0.4643	0.7647	0.5778	0.0054	0.4643	0.7647	0.5778	0.0055
# Processing [SpaCy] |################################| 252/252
# Processing [SpaCy] |################################| 84/84
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	36	21	12	0	0	0.6316	0.7500	0.6857	0.0052	0.6316	0.7500	0.6857	0.0053
# Processing [SpaCy] |################################| 252/252
# Processing [SpaCy] |################################| 84/84
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	37	18	9	0	0	0.6727	0.8043	0.7327	0.0042	0.6727	0.8043	0.7327	0.0044
# Processing [SpaCy] |################################| 252/252
# Processing [SpaCy] |################################| 84/84
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	35	26	20	0	0	0.5738	0.6364	0.6034	0.0051	0.5738	0.6364	0.6034	0.0049
# Processing [SpaCy] |################################| 252/252
# Processing [SpaCy] |################################| 84/84
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	34	15	13	0	0	0.6939	0.7234	0.7083	0.0055	0.6939	0.7234	0.7083	0.0058
#
# # FINAL
# # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
# r_4	181	125	66	0	0	0.5915	0.7328	0.6546	0.0021	0.5915	0.7328	0.6546	0.0022
