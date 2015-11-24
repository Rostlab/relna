import requests

from itertools import chain
from nala.utils.cache import Cacheable

class Swissprot(Cacheable):
    """
    Helper class that accesses the database identifier mapping service from Swissprot.
    and returns a list of 2-tuple with the requested geneid and the corresponding Uniprot ID.
    It defaults to Uniprot ID if the corresponding Swissprot entry is not found.
    """

    def __init__(self):
        super().__init__()
        self.url = 'http://www.uniprot.org/uniprot/?query=database%3A%28type%3Ageneid+id%3A{}%29+AND+reviewed%3A{}&sort=score&format=list'

    def get_uniprotid_for_entrez_geneid(self, list_geneids):
        """
        Get dictionary mapping from { EntrezGeneID : [ UniprotID, ... ]
        :param list_geneids:
        :type list_geneids: [int] or [str] or int or str
        :return: dictionary geneid --> uniprotid-list
        """
        return_dict = {}
        to_be_downloaded = []

        for geneid in list_geneids:
            geneid = str(geneid)
            if geneid in self.cache:
                return_dict[geneid] = self.cache[geneid]
            else:
                to_be_downloaded.append(geneid)

        if len(to_be_downloaded) == 0:
            return return_dict

        for geneid in to_be_downloaded:
            r = requests.get(self.url.format(geneid, 'yes'))
            if r.text=='':
                r = requests.get(self.url.format(geneid, 'no'))
            if r.text!='':
                return_dict[geneid] = [r.text.splitlines()[0]]
                self.cache[geneid] = return_dict[geneid]

        return return_dict
