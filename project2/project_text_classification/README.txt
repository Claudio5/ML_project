Structure of our files

!!! data folder needs to be downloaded in order to make the run.py work !!!
data/ (provided in the following link: https://drive.google.com/open?id=16VZ8valOIURY1Iw5pmCxDhACssU62qq2 )
	- train_neg_full.txt: Provided negative tweets from crowdAI 
	- train_pos_full.txt: Provided positive tweets from CrowdAI
	- crowdai_cleaned_train.csv: Cleaned data which combines the train_pos_full.txt 
					and train_neg_full.txt (data provided by crowdAI)
	- clean_and_merge_train.csv: Merge of stanford and crowdAI data (cleaned version)

pretrained_word2vec/ (provided in the following link: https://drive.google.com/open?id=16VZ8valOIURY1Iw5pmCxDhACssU62qq2 )
	- glove.twitter.27B.50d.txt: Pre-trained word2vec with 50 features 1.2M of tweets
	- glove.twitter.27B.100d.txt: Pre-trained word2vec with 100 features 1.2M of tweets
	- glove.twitter.27B.200d.txt: Pre-trained word2vec with 200 features 1.2M of tweets

./

	- glove.ipynb: Embedding of tweets words with pre-trained word2vec files. 
			Then training a classifier with the results obtained
	
	- cleaned_and_countvec.ipynb: Cleaning of the data (this produces the cleaned data present in data file)
	- run.py: Executable that produces our predictions for the tweets in csv format.


Libraries needed for run.py:
- sklearn
- pandas
- numpy 
- csv

Flow of run.py:
1. The data has been previously cleaned and is directly loaded (both training and testing data). 
   The cleaning part is explained on the report and corresponding code is in jupyter notebook.
2. Classifier and vectorizer are created and trained by the training data.
3. We apply the model to the testing data and output the predictions (predictions.csv) is created in the local folder.