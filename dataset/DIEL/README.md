# Cora Dataset
## download
```console
foo@bar:~/Desktop/implementation_of_papers/dataset/DIEL$ ./downloader.sh
```

## [dataset](http://www.cs.cmu.edu/~lbing/data/emnlp-15-diel/emnlp-15-diel.tar.gz)
This data is used in our EMNLP 2015 paper, with title ��Improving Distant 
Supervision for Information Extraction Using Label Propagation Through 
Lists��. The Freebase seeds are extracted from a snapshot in 2014-04, and
the bipartite graph and features are processed from a corpus downloaded 
from dailymed.nlm.nih.gov which contains 28,590 XML documents.

Feel free to contact the authors for any unclear issue, and please cite our
paper if you use this data in your works.

1 SEEDS (in seeds/)
We extracted entity instances of four types from Freebase, namely disease 
(including medical condition), drug, ingredient, and symptom. These seeds 
are given in different forms as described in the following subsections.

1.1 Original Freebase Seeds (in ./original/)
Here you will find the original form of the extracted seeds from Freebase.
They are basically instances of corresponding types in Freebase:
  - drug.txt: from the type "Drug"
  - disease.txt: from the type "Disease or medical condition"
  - ingredient.txt: from the type "Drug ingredient"
  - symptom.txt: from the type "Symptom"

1.2 Cleaned Freebase Seeds (in ./cleaned/)
As you might observe, the original seeds are very noisy, so we use some 
simple rules to clean them.
  - if a seed contains '(', then seed = seed.substring(0, line.indexOf('('))
  - else 
    - if its length is greater than a threshold, discard it 
    (threshold: 40, 60, 60, 30 for disease, drug, ingredient and symptom)
    - if it contains ',', discard it
  
1.3 Cleaned Single Class Freebase Seeds (in ./cleaned_singleClass/)
Some seeds belong to multiple types. E.g., "headache" belongs to disease
and symptom. Here we give the seeds that belongs to a single type.

1.4 Seed Partitions (in ./runs/)
Here your will see 10 runs, which are used in our EMNLP paper 
(http://www.cs.cmu.edu/~wcohen/postscript/emnlp-2015-baidu.pdf),
each has a random partition of the seeds for the purpose of bias avoidance. 
In each run, the operations are:
  - The single class seeds are firstly split into 50% vs 50%, and named 
    TYPE_devel_50p and TYPE_heldout_eva_50p. 
  - TYPE_devel_50p is split into 80% vs 20%, named 
    TYPE_devel_50p_proppr_seed_forTrainList and 
    TYPE_devel_50p_proppr_seed_forTestList
  - TYPE_devel_50p_proppr_seed_forTrainList is further sampled into subset 
    of different ratios, 0.025, 0.075, 0.125, 0.25, 0.5, 0.75, and 1.
  - Finally, we added back the multi-class seeds back to TYPE_heldout_eva_50p
    and merge the four types to get an evaluation file: coverage_eva_multiAdded

1.5 Seed Label Map
In file: labelMap

2 BIPARTITE GRAPH (in list-graph/)
One type of vertices are list items (ie. NPs) appearing in lists, and the other
type of vertices are the list IDs. Here a list is basically a language pattern
"A, B, and C". We also consider a single NP as a singleton list, containing 
one item. We have 4,464,261 list items, and 4,036,700 lists. 

You will see two files, hasItem.cfacts, and inList.cfacts. One is an inverse 
of the other. Each line is a triple, eg "hasItem s_10000008_16 close_analogue��,
which tells the list s_10000008_16 has an item close_analogue.

3 FEATRUE (in feature/)
Here you will find the raw features of each list, in the form of tokens. 
An example line is: "s_10000008_16   bw=although bw=not ... ", where the list 
ID is followed with the feature tokens.
We have 5 types of features: 
  - bow_context.tok_feat: each token feature is in the form of "bw=TOK", where 
    TOK is a word in the sentence containing the list. The words from the list
    itself are excluded.
  - close_context.tok_feat: (THERE IS A BUG FOR THIS FEATURE, which affect all
    compared pipelines in our paper. A corrected data will be released later)
    We have eight types: l3=, l2= l1=, r3=, r2=, r1=, lt=, and rt=. lX (rX) 
    means leftXgram (rightXgram) on the left (right) of  the list. lt (rt) 
    means a single token in l3 (r3). Therefore, we have at most 12 features.
  - dep.tok_feat: from the dependence parseing, we got the verb which is the 
    closest ancestor of the head of the NP, all modifiers of this verb, and the
    path to this verb. For a list, the dependency features are computed 
    relative to the head of the list.
  - list_bow.tok_feat: the individual words in the list.
  - pre_suffix.tok_feat: prefix and suffix (length 3 and 4) of each word in 
    the list.
