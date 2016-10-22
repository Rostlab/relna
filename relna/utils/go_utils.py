import requests
from nalaf.utils.cache import Cacheable

class GOTerms(Cacheable):
    """
    Helper class that accesses the database identifier mapping service from Uniprot.
    and returns a list of 2-tuple with the requested geneid and the corresponding Uniprot ID.
    """
    def __init__(self):
        super().__init__()
        self.url = 'http://www.uniprot.org/uniprot/{}.txt'


    def get_goterms_for_uniprot_id(self, list_uniprotids):
        """
        Get dictionary mapping from { UniprotID : [ GOTerms, ... ]
        :param list_geneids:
        :type list_geneids: [int] or [str] or int or str
        :return: dictionary uniprotid --> goterms-list
        """
        return_dict = {}
        to_be_downloaded = []

        for uniprotid in list_uniprotids:
            if uniprotid in self.cache:
                return_dict[uniprotid] = self.cache[uniprotid]
            else:
                to_be_downloaded.append(uniprotid)

        if len(to_be_downloaded) == 0:
            return return_dict

        for uniprotid in to_be_downloaded:
            return_dict[uniprotid] = []
            r = requests.get(self.url.format(uniprotid))
            for line in r.text.splitlines():
                if line.startswith("DR   GO;"):
                    startIndex = line.find('GO:')
                    endIndex = startIndex+10
                    return_dict[uniprotid].append(line[startIndex:endIndex])

        return return_dict
