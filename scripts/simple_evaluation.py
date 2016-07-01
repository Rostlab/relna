from nalaf.utils.readers import HTMLReader
from nalaf.utils.annotation_readers import AnnJsonAnnotationReader
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator
from relna.utils import PRO_CLASS_ID, MUT_CLASS_ID, PRO_REL_MUT_CLASS_ID
from relna.structures.relation_pipelines import RelationExtractionPipeline
from relna.learning.svmlight import SVMLightTreeKernels
from relna.preprocessing.parsers import SpacyParser, BllipParser
from spacy.en import English
from relna.learning.taggers import TranscriptionFactorTagger
from relna.learning.taggers import RelnaRelationExtractor

relna = False
use_tk = False

if use_tk:
    svm_folder = '/usr/local/manual/svm-light-TK-1.2.1/'
    nlp = English(entity=False)
    parser = SpacyParser(nlp, constituency_parser=True)
    parser = BllipParser()
else:
    svm_folder = '/usr/local/manual/bin/'
    parser = None

if relna:
    # Relna
    dataset_folder_html = '/Users/jmcejuela/Work/hck/relna/resources/corpora/relna/corrected/'
    dataset_folder_annjson = dataset_folder_html
    rel_type = 'r_4'
else:
    # LocText
    dataset_folder_html = '/Users/jmcejuela/Work/hck/relna/resources/corpora/LocText/LocText_plain_html/pool/'
    dataset_folder_annjson = '/Users/jmcejuela/Work/hck/relna/resources/corpora/LocText/LocText_master_json/pool/'
    rel_type = 'r_5'

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

k = 5

for fold in range(k):
    training, validation, test = dataset.cv_kfold_split(k, fold)
    tagger.tag(validation)

    r = evaluator.evaluate(validation)
    print(r)

print("\n\n\n\n\n")

dataset = read_dataset()

evaluator = DocumentLevelRelationEvaluator(rel_type=rel_type, match_case=False)

pipeline = RelationExtractionPipeline('e_1', 'e_2', rel_type, parser=parser)

for fold in range(k):

    training, validation, test = dataset.cv_kfold_split(k, fold, validation_set=False)
    validation = test

    pipeline.execute(training, train=True)
    feature_set = pipeline.feature_set

    # get the predictions
    svmlight = SVMLightTreeKernels(svm_folder, use_tree_kernel=use_tk)
    svmlight.create_input_file(training, 'train', feature_set)
    svmlight.learn()

    pipeline.execute(validation, train=False, feature_set=feature_set)
    svmlight.create_input_file(validation, 'test', feature_set)
    svmlight.tag(mode='test')
    svmlight.read_predictions(validation)

    results = evaluator.evaluate(validation)
    print(results)
