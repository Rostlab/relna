from nalaf.utils.readers import HTMLReader
from nalaf.utils.annotation_readers import AnnJsonAnnotationReader
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from relna.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.learning.svmlight import SVMLightTreeKernels
from relna.preprocessing.parsers import SpacyParser
from spacy.en import English
# from relna.learning.taggers import TranscriptionFactorTagger
# from relna.learning.taggers import RelnaRelationExtractor
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
    svm_folder = '/usr/local/manual/svm-light-TK-1.2.1/'
    nlp = English(entity=False)
    parser = SpacyParser(nlp, constituency_parser=True)
else:
    svm_folder = '/usr/local/manual/bin/'
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

    pipeline.execute(training, train=True)
    feature_set = pipeline.feature_set

    # get the predictions
    svmlight = SVMLightTreeKernels(svm_folder, use_tree_kernel=args.use_tk)
    svmlight.create_input_file(training, 'train', feature_set)
    svmlight.learn()

    pipeline.execute(validation, train=False, feature_set=feature_set)
    svmlight.create_input_file(validation, 'test', feature_set)
    svmlight.tag(mode='test')
    svmlight.read_predictions(validation)

    results = evaluator.evaluate(validation)
    merged.append(results)
    print(results)

print("\n# FINAL")
ret = Evaluations.merge(merged)
print(ret)
