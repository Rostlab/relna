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
import math

def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='Simple-evaluate relna corpus corpus')

    parser.add_argument('--corpus', default="relna", choices=["relna"])
    parser.add_argument('--use_tk', default=False, action='store_true')
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--use_full_corpus', default=False, action='store_true')

    args = parser.parse_args(argv)

    print(args)

    return args


def test_whole_with_defaults(argv=None):
    argv = [] if argv is None else argv
    args = parse_arguments(argv)

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
        dataset_folder_html = './resources/corpora/relna/corrected/'
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

    rel_evaluation = ret(rel_type).compute(strictness="exact")

    EXPECTED_F = 0.34506089309878213
    EXPECTED_F_SE = 0.0017486985851191787

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F)
    assert math.isclose(rel_evaluation.f_measure_SE, EXPECTED_F_SE, rel_tol=0.1)

    print("\n\n\n")

    if (args.use_full_corpus):
        dataset = read_dataset()
    else:
        dataset, _ = read_dataset().percentage_split(0.1)

    feature_generators = RelnaRelationExtractor.default_feature_generators('e_1', 'e_2')
    pipeline = RelationExtractionPipeline('e_1', 'e_2', rel_type, parser=parser, feature_generators=feature_generators)

    evaluator = DocumentLevelRelationEvaluator(rel_type=rel_type, match_case=False)

    print("# FOLDS")
    merged = []
    for fold in range(k):

        training, validation, test = dataset.cv_kfold_split(k, fold, validation_set=(not args.use_test_set))
        if args.use_test_set:
            validation = test

        # Learn
        pipeline.execute(training, train=True)
        svmlight = SVMLightTreeKernels(svmlight_dir_path=svm_folder, use_tree_kernel=args.use_tk)
        instancesfile = svmlight.create_input_file(training, 'train', pipeline.feature_set, minority_class=1, majority_class_undersampling=0.4)
        svmlight.learn(instancesfile)

        # Predict & Read predictions
        pipeline.execute(validation, train=False)
        instancesfile = svmlight.create_input_file(validation, 'test', pipeline.feature_set)
        predictionsfile = svmlight.tag(instancesfile)
        svmlight.read_predictions(validation, predictionsfile, threshold=-0.1)  # CAUTION! previous relna svm_light had the threshold of prediction at '-0.1' -- nalaf changed it to 0 (assumed to be correct) -- This does change the performance and actually reduce it in this example

        results = evaluator.evaluate(validation)
        merged.append(results)
        print(results)

    print("\n# FINAL")
    ret = Evaluations.merge(merged)
    print(ret)

    rel_evaluation = ret(rel_type).compute(strictness="exact")

    if (args.use_full_corpus):
        # Beware that performance depends a lot on the undersampling and svm threshold
        EXPECTED_F = 0.7008
        EXPECTED_F_SE = 0.0018
    else:
        # I even achieved this when spacy was not really parsing: 0.6557
        EXPECTED_F = 0.6452
        EXPECTED_F_SE = 0.0055

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)


if __name__ == "__main__":
    import sys
    test_whole_with_defaults(sys.argv[1:])
