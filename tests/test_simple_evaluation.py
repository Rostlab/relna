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
    parser.add_argument('--corpus_percentage', default=0.1, type=float, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--minority_class', type=int, default=1, choices=[-1, 1])
    parser.add_argument('--majority_class_undersampling', type=float, default=0.4)
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)
    parser.add_argument('--use_tk', default=False, action='store_true')

    args = parser.parse_args(argv)

    print(args)

    return args


def test_whole_with_defaults(argv=None):
    argv = [] if argv is None else argv
    args = parse_arguments(argv)

    if args.use_tk:
        nlp = English(entity=False)
        parser = SpacyParser(nlp, constituency_parser=True)
    else:
        parser = None

    if args.corpus == "relna":
        dataset_folder_html = './resources/corpora/relna/corrected/'
        dataset_folder_annjson = dataset_folder_html
        e_id_1 = 'e_1'
        e_id_2 = 'e_2'
        r_id = 'r_4'


    def read_dataset():
        dataset = HTMLReader(dataset_folder_html).read()
        AnnJsonAnnotationReader(
                dataset_folder_annjson,
                read_only_class_id=None,
                read_relations=True,
                delete_incomplete_docs=False).annotate(dataset)

        return dataset


    def test_baseline():

        dataset = read_dataset()
        tagger = StubSameSentenceRelationExtractor(e_id_1, e_id_2, r_id)
        evaluator = DocumentLevelRelationEvaluator(rel_type=r_id, match_case=False)

        print("# FOLDS")
        merged = []
        for fold in range(args.k_num_folds):
            training, validation, test = dataset.cv_kfold_split(args.k_num_folds, fold, validation_set=(not args.use_test_set))
            if args.use_test_set:
                validation = test

            tagger.tag(validation)

            r = evaluator.evaluate(validation)
            merged.append(r)
            print(r)

        print("\n# FINAL")
        ret = Evaluations.merge(merged)
        print(ret)

        rel_evaluation = ret(r_id).compute(strictness="exact")

        EXPECTED_F = 0.5621
        EXPECTED_F_SE = 0.0021

        assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure


    def test_relna():

        if (args.corpus_percentage == 1.0):
            dataset = read_dataset()
        else:
            dataset, _ = read_dataset().percentage_split(args.corpus_percentage)


        def train(training_set):
            feature_generators = RelnaRelationExtractor.default_feature_generators(e_id_1, e_id_2)
            pipeline = RelationExtractionPipeline(e_id_1, e_id_2, r_id, parser=parser, tokenizer=TmVarTokenizer(), feature_generators=feature_generators)

            pipeline.execute(training_set, train=True)
            svmlight = SVMLightTreeKernels(use_tree_kernel=args.use_tk)
            instancesfile = svmlight.create_input_file(training_set, 'train', pipeline.feature_set, minority_class=args.minority_class, majority_class_undersampling=args.majority_class_undersampling)
            svmlight.learn(instancesfile)

            def annotator(validation_set):
                pipeline.execute(validation_set, train=False)
                instancesfile = svmlight.create_input_file(validation_set, 'predict', pipeline.feature_set)
                predictionsfile = svmlight.tag(instancesfile)
                # CAUTION! previous relna svm_light had the threshold of prediction at '-0.1' -- nalaf changed it to 0 (assumed to be correct) -- This does change the performance and actually reduce it in this example
                svmlight.read_predictions(validation_set, predictionsfile, threshold=-0.1)
                return validation_set

            return annotator

        evaluator = DocumentLevelRelationEvaluator(rel_type=r_id, match_case=False)

        evaluations = Evaluations.cross_validate(train, dataset, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
        print(evaluations)

        rel_evaluation = evaluations(r_id).compute(strictness="exact")


        if (args.corpus_percentage == 1.0):
            # Beware that performance depends a lot on the undersampling and svm threshold
            EXPECTED_F = 0.7008
            EXPECTED_F_SE = 0.0018
        elif (args.corpus_percentage == 0.1):
            # I even achieved this when spacy was not really parsing: 0.6557
            EXPECTED_F = 0.6452
            EXPECTED_F_SE = 0.0055
        else:
            # This is not to be tested and will fail
            EXPECTED_F = 0.5
            EXPECTED_F_SE = 0.00001

        assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    test_baseline()
    test_relna()


if __name__ == "__main__":
    import sys
    test_whole_with_defaults(sys.argv[1:])
