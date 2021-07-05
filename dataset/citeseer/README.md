# CiteSeer Dataset
## download
```console
foo@bar:~/Desktop/implementation_of_papers/dataset/citeseer$ ./conda_downloader.sh
```

## [dataset](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html)
The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.

This directory contains the a selection of the CiteSeer dataset.

These papers are classified into one of the following six classes:
			Agents
			AI
			DB
			IR
			ML
			HCI

The papers were selected in a way such that in the final corpus every paper cites or is cited by atleast one other paper. There are 3312 papers in the whole corpus. 

After stemming and removing stopwords we were left with a vocabulary of size 3703 unique words. All words with document frequency less than 10 were removed.


THE DIRECTORY CONTAINS TWO FILES:

The .content file contains descriptions of the papers in the following format:

		<paper_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. Finally, the last entry in the line contains the class label of the paper.

The .cites file contains the citation graph of the corpus. Each line describes a link in the following format:

		<ID of cited paper> <ID of citing paper>

Each line contains two paper IDs. The first entry is the ID of the paper being cited and the second ID stands for the paper which contains the citation. The direction of the link is from right to left. If a line is represented by "paper1 paper2" then the link is "paper2->paper1". 