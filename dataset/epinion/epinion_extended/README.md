# Epinion_extended dataset

- Manually download dataset from [here](http://www.trustlet.org/epinions.html)


## dataset README
This dataset was given directly by Epinions staff to Paolo Massa. As a consequence, the dataset contains also the distrust lists (which users are distrusted by which users) that is not shown on the site but kept private. Note that it is not a tipical collaborative filtering dataset, since the ratings are about the articles and not about items: the ratings represent how much a certain user rates a certain textual article written by an other user, i.e. a review.

The dataset contains

* ~132,000 users, who issued
* 841,372 statements (717,667 trusts and 123,705 distrusts).
* ~85,000 users received at least one statement.

Users and Items are represented by anonimized numeric identifiers.

The dataset consists of 3 files.

user_rating.txt.gz (4.7 Megabytes): Trust is the mechanism by which the user makes a statement that he likes the content or the behavior of particular user and would like to see more of what the users does in the site. Distrust is the opposite of the trust in which the user says that they do want to see lesser of the operations performed by that user.

Column Details:

MY_ID This stores Id of the member who is making the trust/distrust statement
OTHER_ID The other ID is the ID of the member being trusted/distrusted
VALUE Value = 1 for trust and -1 for distrust
CREATION It is the date on which the trust was made
mc.txt.gz (15 Megabytes): Each article is written by a user.

Column Details:

CONTENT_ID The object ID of the article.
AUTHOR_ID The ID of the user who wrote the article
SUBJECT_ID The ID of the subject that the article is supposed to be about
rating.txt (684 Megabytes): Ratings are quantified statements made by users regarding the quality of a content in the site. Ratings is the basis on which the contents are sorted and filtered.

Column Details:-

OBJECT_ID The object ID is the object that is being rated. The only valid objects at the present time are the content_id of the member_content table. This means that at present this table only stores the ratings on reviews and essays
MEMBER_ID Stores the id of the member who is rating the object
RATING Stores the 1-5 (1- Not helpful , 2 - Somewhat Helpful, 3 - Helpful 4 - Very Helpful 5- Most Helpful) rating of the object by member [There are some 6s, treat them as 5]
STATUS The display status of the rating. 1 :- means the member has chosen not to show his rating of the object and 0 meaning the member does not mind showing his name beside the rating.
CREATION The date on which the member first rated this object
LAST_MODIFIED The latest date on which the member modified his rating of the object
TYPE If and when we allow more than just content rating to be stored in this table, then this column would store the type of the object being rated.
VERTICAL_ID Vertical_id of the review.
How to use these files?

Just download the txt.gz files on your hard disk. Then run from the command line of your GNU/Linux shell:

gunzip name_of_file.txt.gz 
Some people reported that under Windows the files seems to be doubly zipped.

When you unzip the files, you'll get a .txt file which is not really a text file. It's still a zip file. Change the extension to .zip and unzip the file again. Then you are done. Let me know if you have any problem.

If you use this dataset, you might want to cite one of the following papers:
Trustlet, open research on trust metrics. P Massa, K Souren, M Salvetti, D Tomasoni. Scalable Computing: Practice and Experience 9 (4)
or
Trust-aware recommender systems. P Massa, P Avesani. Proceedings of the 2007 ACM conference on Recommender systems, 17-24
