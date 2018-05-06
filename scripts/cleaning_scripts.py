def clean_tweet_chars(tweet):
	s = ''
	for c in tweet:
		t = ord(c)
		if (t>=65 and t<=90) or (t>=97 and t<=122) or (t==32) or (t==39):
			s += c
		else:
			s += ' '
	s = s.strip()
	return s