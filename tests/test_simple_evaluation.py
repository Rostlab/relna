from nalaf.utils.readers import HTMLReader
from nalaf.utils.annotation_readers import AnnJsonAnnotationReader
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.preprocessing.tokenizers import TmVarTokenizer, NLTK_TOKENIZER
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
    parser.add_argument('--corpus_percentage', default=0.5, type=float, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--minority_class', type=int, default=1, choices=[-1, 1])
    parser.add_argument('--majority_class_undersampling', type=float, default=0.4)
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)
    parser.add_argument('--use_tk', default=False, action='store_true')

    args = parser.parse_args(argv)

    if args.corpus == "relna":
        args.dataset_folder_html = './resources/corpora/relna/corrected/'
        args.dataset_folder_annjson = args.dataset_folder_html
        args.e_id_1 = 'e_1'
        args.e_id_2 = 'e_2'
        args.r_id = 'r_4'

    print(args)

    return args


def read_dataset(args):

    dataset = HTMLReader(args.dataset_folder_html).read()
    AnnJsonAnnotationReader(
            args.dataset_folder_annjson,
            read_only_class_id=None,
            read_relations=True,
            delete_incomplete_docs=False,
            raise_exception_on_incosistencies=False).annotate(dataset)

    return dataset


def test_baseline(argv=None):
    argv = [] if argv is None else argv
    args = parse_arguments(argv)

    dataset = read_dataset(args)

    # Computation(precision=0.389351081530782, precision_SE=0.0021024361502353277, recall=0.9790794979079498, recall_SE=0.0007302523934357751, f_measure=0.5571428571428572, f_measure_SE=0.0021357013961776057)
    # Full corpus
    EXPECTED_F = 0.5571
    EXPECTED_F_SE = 0.0021

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(args.e_id_1, args.e_id_2, args.r_id).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=args.r_id)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, dataset, evaluator, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(args.r_id).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure


def test_relna(argv=None):
    argv = [] if argv is None else argv
    args = parse_arguments(argv)

    if (args.corpus_percentage == 1.0):
        dataset = read_dataset(args)
        # Beware that performance depends a lot on the undersampling and svm threshold
        EXPECTED_F = 0.6979
        EXPECTED_F_SE = 0.0019

    else:
        dataset, _ = read_dataset(args).percentage_split(args.corpus_percentage)

        if (args.corpus_percentage == 0.5):
            EXPECTED_F = 0.6094
            EXPECTED_F_SE = 0.0029
        else:
            # This is not to be tested and will fail
            EXPECTED_F = 0.5
            EXPECTED_F_SE = 0.00001

    if args.use_tk:
        nlp = English(entity=False)
        parser = SpacyParser(nlp, constituency_parser=True)
    else:
        parser = None

    def train(training_set):
        feature_generators = RelnaRelationExtractor.default_feature_generators(args.e_id_1, args.e_id_2)
        pipeline = RelationExtractionPipeline(args.e_id_1, args.e_id_2, args.r_id, parser=parser, tokenizer=TmVarTokenizer(), feature_generators=feature_generators)

        pipeline.execute(training_set, train=True)

        # CAUTION! previous relna svm_light had the threshold of prediction at '-0.1' -- nalaf changed it to 0 (assumed to be correct) -- This does change the performance and actually reduce it in this example
        # http://svmlight.joachims.org For classification, the sign of this value determines the predicted class -- CAUTION, relna (Ashish), had it set before to exactly: '-0.1' (was this a bug or a conscious decision to move the threshold of classification?)
        # See more information in: https://github.com/Rostlab/relna/issues/21
        svmlight = SVMLightTreeKernels(classification_threshold=-0.1, use_tree_kernel=args.use_tk)
        instancesfile = svmlight.create_input_file(training_set, 'train', pipeline.feature_set, minority_class=args.minority_class, majority_class_undersampling=args.majority_class_undersampling)
        svmlight.learn(instancesfile, c=0.5)

        def annotator(validation_set):
            pipeline.execute(validation_set, train=False)
            instancesfile = svmlight.create_input_file(validation_set, 'predict', pipeline.feature_set)
            predictionsfile = svmlight.classify(instancesfile)

            svmlight.read_predictions(validation_set, predictionsfile)
            return validation_set

        return annotator

    evaluator = DocumentLevelRelationEvaluator(rel_type=args.r_id)
    evaluations = Evaluations.cross_validate(train, dataset, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
    rel_evaluation = evaluations(args.r_id).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)


if __name__ == "__main__":
    import sys
    real_args = sys.argv[1:]
    test_baseline(real_args)
    test_relna(real_args)
