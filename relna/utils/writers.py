from itertools import chain
import sys
import os
from relna.utils import MUT_CLASS_ID, PRO_CLASS_ID


class RelnaConsoleWriter:
    """
    Writes the predicted or true annotations to console, along with the true
    and predicted relations.
    """
    def __init__(self, color=True):
        self.color = self.__supports_color() if color else color
        self.protein_color_start = '\033[42m'
        self.mutation_color_start = '\033[41m'
        self.end_color = '\033[0m'
        pass

    def write(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset()
        """
        for doc_id, doc in dataset.documents.items():
            print('DOCUMENT: {}'.format(doc_id))
            for part_id, part in doc.parts.items():
                print('PART {}'.format(part_id))
                self.___print_part(part)

    def ___print_part(self, part):
        if self.color:
            text = part.text
            total_offset = 0
            for ann in sorted(chain(part.predicted_annotations, part.annotations), key=lambda x: x.offset):
                color = self.mutation_color_start if ann.class_id == MUT_CLASS_ID else self.protein_color_start
                text = text[:ann.offset+total_offset] + color + text[ann.offset+total_offset:]
                total_offset += 5
                text = text[:ann.offset+len(ann.text)+total_offset] + self.end_color + text[ann.offset+len(ann.text)+total_offset:]
                total_offset += 4
            print(text)
            print()
        else:
            padding = len(str(len(part.text)))
            print(part.text)
            print('ANNOTATIONS')
            for ann in sorted(chain(part.predicted_annotations, part.annotations), key=lambda x: x.offset):
                if ann.class_id == MUT_CLASS_ID:
                    print('Mutation {0: <{pad}} {1: <{pad}} {2}'
                          .format(ann.offset, ann.offset+len(ann.text), ann.text, pad=padding))
                elif ann.class_id == PRO_CLASS_ID:
                    print('GGP      {0: <{pad}} {1: <{pad}} {2} {3}'
                          .format(ann.offset, ann.offset+len(ann.text), ann.text, ann.normalisation_dict, pad=padding))
        print('RELATIONS')
        for rel in chain(part.relations, part.predicted_relations):
            print('{} ---> {}'.format(rel.text1, rel.text2))
        print()

    @staticmethod
    def __supports_color():
        """
        Returns True if the running system's terminal supports color, and False
        otherwise.
        """
        plat = sys.platform
        supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        if not supported_platform or not is_a_tty:
            return False
        return True
