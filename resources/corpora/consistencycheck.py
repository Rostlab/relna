import os
import json
from pprint import pprint

tagged_dir = 'tagged'
untagged_dir = 'process'
process = 'process'

tagged_files = os.listdir(tagged_dir)
need_processing = []
for file_name in tagged_files:
    with open(os.path.join(tagged_dir, file_name)) as f:
        tagged = json.load(f)
    pub_id = tagged['sources'][0]['id']
    with open(os.path.join(untagged_dir, file_name)) as g:
        untagged = json.load(g)
        try:
            for entity in untagged['entities']:
                if entity['normalizations'] == {}:
                    print (pub_id, entity['start'][0]['text'], entity['start'[]])
                    need_processing.append(pub_id)
        except AssertionError:
            pass
print set(need_processing), len(set(need_processing))
    #     for i in range(len(tagged['entities'])):
    #         entity1 = tagged['entities'][i]
    #         offset = entity1['offsets'][0]['start']
    #         text = entity1['offsets'][0]['text']
    #         for entity2 in untagged['entities']:
    #             if entity2['offsets'][0]['start']==offset and entity2['offsets'][0]['text']==text and entity2['part']==entity1['part']:
    #                 entity1['normalizations'] = entity2['normalizations']
    #                 try:
    #                     entity1['normalizations']['n_7']['confidence']['who']=['user:Ashish']
    #                 except KeyError:
    #                     pass
    #                 entity2[u'fields'] = {}
    #                 entity1['confidence']['who'][0] = u'ml:GNormPlus'
    #                 entity1['confidence']['prob'] = 1.0
    #                 entity2['classId'] = entity1['classId']
    #                 entity1['confidence']['state'] = ''
    #                 # tagged['entities'][j] = entity2
    #                 try:
    #                     assert entity2 == entity1
    #                     tagged['entities'][i] = entity1
    #                 except AssertionError:
    #                     print ('TagTog version - MAINTAIN original')
    #                     pprint (entity1)
    #                     print
    #                     print ('Local Version')
    #                     pprint (entity2)
    #                     raise AssertionError("Entities not equal.")
    # with open(os.path.join(process, file_name), 'w') as fw:
    #     fw.write(json.dumps(tagged, sort_keys=True, indent=2, separators=(',', ': ')))
