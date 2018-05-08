def clean_tweet_chars(text):
	s = ''
	for c in text:
		t = ord(c)
		if (t>=65 and t<=90) or (t>=97 and t<=122) or (t==32) or (t==39):
			s += c
		else:
			s += ''
	s = s.strip()
	return s

def clean_text_per_line_from_tsv(file_name, tweet_col):
	f = open(file_name, 'r')
	o = open(file_name+"_tweets", 'w')
	for line in f:
		cols = line.split("\t")
		tweet = cols[tweet_col]
		tweet = tweet.strip(" \n")
		c_tweet = clean_tweet_chars(tweet)
		o.write(c_tweet + "\n")