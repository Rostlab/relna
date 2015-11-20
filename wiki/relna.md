
Corpus
======
Relna
-------
Relna is a corpus of interactions between transcription factors and gene or gene products (GGP). Transcriptional regulation is one of the fundamental ways in which gene regulation occurs across several organisms. **Relna** seeks to annotate functional relationships between 

 - transcription factors and the genes they transcribe (transcription factor _transcribes_ gene)
 - transcription factors and proteins that modify the behavior of these
   transcription factors (protein _modifies_ transcription factor)

to make identification of such interactions possible from natural text.

A growing amount of research attributes dysfunctional transcription factors to several genetic irregularities and diseases (several oncogenes are transcription factors[^cancertherapy]). Additionally, several tumor suppressor genes are also transcription factors. Understanding their interactions with different genes and proteins could help design better cures.

Corpus Details
-------------
**Relna** consists of 140 abstracts semi-automatically annotated for transcription factors (TF) and gene or gene products (GGP). Both the TF and GGPs are normalized to both their **Entrez Gene ID** and **UniProt ID**. The relations between TFs and GGPs were manually annotated using TagTog[^tagtog].

Document Selection
------------------------

 1. **SwissProt** was filtered for proteins from _Homo sapiens_ and having the Gene Ontology Term [`GO:0003700`](http://amigo.geneontology.org/amigo/term/GO:0003700) - (_transcription factor activity, sequence-specific DNA binding_). This gives a list of over 1100 proteins. 
 2. For each of these proteins, we obtain a list of publications that have cited this specific protein. 
 3. From among the list publications, we further filter those papers that have been cited for **"INTERACTION WITH"**.
 4. This forms the _base corpus_ of 1461 abstracts, with no entity mentions or relations.

Named Entity Recognition
-------------------------------
A random sample of 140 abstracts is picked from the _base corpus_ for named entity recognition. We use state of the art GNormPlus[^gnormplus] for tagging our abstracts with gene or gene product names. GNormPlus currently annotates genes, gene families, species and domain-motifs. GNormPlus was accessed through its API[^gnormplusapi] and returns all results in the PubTator[^pubtator] format. GNormPlus also performs gene normalization against the Entrez Gene Database[^entrez]. The information of gene families, species and domain-motifs is ignored and **relna** focuses solely on genes (or gene products).

Each gene or gene product annotated by GNormPlus is cross-referenced with SwissProt using the Entrez Gene ID. It is important to note here that this process can be slightly error prone for two reasons:

1. Firstly, gene normalization is not entirely accurate - for instance, normalization sometimes loses information about species and we have realized that some Gene IDs are from _Mus musculus_. In a very small number of cases, the normalized protein may not match the original.
2.  Secondly, there is no one-to-one mapping between Entrez Gene IDs and UniProt IDs, indeed it is actually a one-to-many mapping. While this problem is alleviated slightly by limiting our database to SwissProt, there are still a few cases where one-to-many mappings exist. In these cases, we naively select the first result from UniProt.

After obtaining the UniProt IDs, we check each protein for the presence of the GO Term _GO:0003700_[^goterm], which indicates transcription factor activity. All gene or gene products having this term are labeled transcription factors, and those that do not are left as GGPs.

In the final step, PubTator files are converted to the TagTog[^tagtogweb] format (_.html and .ann.json_) using a Python library PubTator2Anndoc[^converter]. The TagTog format can be accessed here[^tagtogwiki].

Annotation Guidelines
---------------------------
### Entities
* **Entity Types**: We annotate entities of the type Transcription Factor and Transcription Factor-binding.

* **Normalization**: An entity is normalizable if it can be looked up in a standard database.
    * Entities must be searchable in NCBI Gene DB and must have a single unique Gene ID or equivalently, a unique Swissprot ID (UniProt contains un-reviewed proteins, that makes it difficult to normalize).

* **Entity Boundaries**: Often GNormPlus misrepresents boundaries. For example, an entity may be incorrectly tagged as `Estrogen receptor (ER`. This is corrected to `Estrogen receptor` and `ER` while reviewed tags.

* **Rule of acronyms**: Wherever the full name is annotated, also annotate the abbreviation / acronym. For instance, In `Androgen receptor (AR)`, `Androgen receptor` and `AR` are both annotated.

* **Unidentified Entities**: If an entity is recognized at one place, and not at the other, annotate all mentions of the entity.

### Relations
* We are looking for physical interactions between TF GGPs and TF-binding GGPs. This gives us two broad categories of relationships.
    * TF-binding proteins that regulate / modify / co-activate other transcription factors.
    * Genes that are transcribed by the transcription factor.
* Do **NOT** annotate relations that do not imply physical binding or interaction.  

#### _Meaningful Relation_ (definition)

* A relation is meaningful if either of following two conditions is true:
      1. We can find the written _words_ used by the author that implies that there can be a relation (in this case, the relation is not inferred):
          * Example: In this work, we investigate the phosphorylation of the N-terminal heterodimerization (PAS) domain of **HIF-1alpha** and identify Ser247 as a major site of in vitro modification by **casein kinase 1delta** (**CK1delta**). Annotate the relation between HIF-1alpha and casein kinase 1delta as the word _modify_ clearly indicates a physical relation.
      2. We can **infer** the relation from the context and it can be verified from UniProt:
          * Example: We further showed that **CCAR1** is required for recruitment of **AR**. While the relation isn't explicit, the sentence indicates that CCAR1 _possibly_ interacts with AR. In these situations, the UniProt entry for AR (Androgen Receptor) was checked to verify the relation between the two entities. Only then was the relation annotated.

#### General rules for relations:

* Always if you relate the long name (or abbreviation) , do relate the abbreviation (or, respectively, long name) as well .
* Names almost equal but still spelt slightly differently should be considered as "different entities" for annotation purposes. That is, when considering relations, you should relate both entities to whatever is needed.

Relna Corpus Statistics
---------------------------

![Relna Corpus Statistics](https://plot.ly/~ashish.baghudana/15.png)

![Relna Corpus Statistics](https://plot.ly/~ashish.baghudana/18.png)

![Relna Corpus Statistics](https://plot.ly/~ashish.baghudana/20.png)


[^cancertherapy]: Libermann TA, Zerbini LF. [Targeting transcription factors for cancer gene therapy.](http://www.ncbi.nlm.nih.gov/pubmed/16475943) Curr Gene Ther. 2006;6(1):17-33.

[^tagtog]: Cejuela JM, Mcquilton P, Ponting L, et al. [tagtog: interactive and text-mining-assisted annotation of gene mentions in PLOS full-text articles.](http://www.ncbi.nlm.nih.gov/pubmed/24715220) Database (Oxford). 2014;2014:bau033.

[^goterm]: [Gene Ontology 0003700 - transcription factor activity, sequence-specific DNA binding](http://amigo.geneontology.org/amigo/term/GO:0003700)

[^gnormplus]: Wei CH, Kao HY, Lu Z. [GNormPlus: An Integrative Approach for Tagging Genes, Gene Families, and Protein Domains.](http://www.hindawi.com/journals/bmri/2015/918710/) Biomed Res Int. 2015;2015:918710.

[^gnormplusapi]: Available at: http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/. Accessed November 19, 2015.

[^pubtator]: Available at: http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html. Accessed November 19, 2015.

[^entrez]: Available at: http://www.ncbi.nlm.nih.gov/gene. Accessed November 19, 2015.

[^converter]: Available at: https://github.com/ashishbaghudana/PubTator2Anndoc. Accessed November 19, 2015.

[^tagtogweb]: Available at: http://tagtog.net. Accessed November 19, 2015.

[^tagtogwiki]: Available at: https://github.com/tagtog/tagtog-doc/wiki/tagtog-document-formats. Accessed November 19, 2015.