# Epinion dataset

- Manually download dataset from [here](http://www.trustlet.org/epinions.html)

## dataset README
The dataset was collected by Paolo Massa in a 5-week crawl (November/December 2003) from the Epinions.com Web site.

The dataset contains

* 49,290 users who rated a total of
* 139,738 different items at least once, writing
* 664,824 reviews and
* 487,181 issued trust statements.
Users and Items are represented by anonimized numeric identifiers.

The dataset consists of 2 files.

ratings_data.txt.bz2 (2.5 Megabytes): it contains the ratings given by users to items.

Every line has the following format:

user_id item_id rating_value
For example,

23 387 5
represents the fact "user 23 has rated item 387 as 5"

Ranges:

user_id is in [1,49290]
item_id is in [1,139738]
rating_value is in [1,5]
trust_data.txt.bz2 (1.7 Megabytes): it contains the trust statements issued by users.

Every line has the following format:

source_user_id target_user_id trust_statement_value
For example, the line

22605 18420 1
represents the fact "user 22605 has expressed a positive trust statement on user 18420"

Ranges:

source_user_id and target_user_id are in [1,49290]
trust_statement_value is always 1 (since in the dataset there are only positive trust statements and not negative ones (distrust)).
Note: there are no distrust statements in the dataset (block list) but only trust statements (web of trust), because the block list is kept private and not shown on the site.

The data were collected using a crawler, written in Perl.
It was the first program I (Paolo Massa) ever wrote in Perl (and an excuse for learning Perl) so the code is probably very ugly. Anyway I release the code under the GNU Generic Public Licence (GPL) so that other people might be use the code if they so wish.
epinionsRobot_pl.txt is the version I used, this version parses the HTML and saves minimal information as perl objects. Later on, I saw this was not a wise choice (for example, I didn't save demographic information about users which might have been useful for testing, for example, is users trusted by user A comes from the same city or region). So later on I created a version that saves the original HTML pages (epinionsRobot_downloadHtml_pl.txt) but I didn't test it. Feel free to let me know if it works. Both Perl files are released under GNU Generic Public Licence (GPL), see first lines of the files.

If you use this dataset, you might want to cite one of the following papers:
Trustlet, open research on trust metrics. P Massa, K Souren, M Salvetti, D Tomasoni. Scalable Computing: Practice and Experience 9 (4)
or
Trust-aware recommender systems. P Massa, P Avesani. Proceedings of the 2007 ACM conference on Recommender systems, 17-24
