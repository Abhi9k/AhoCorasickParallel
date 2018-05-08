def generate_records_of_length(length, file_name):
	f = open(file_name, 'r')
	o = open(file_name + "_" + str(length), 'w')
	s = 0
	rec = ""
	for line in f:
		line = line.strip(" \n")
		rec += line
		s += len(line)
		if s >= length:
			rec = rec[:length-1]
			rec = rec + "\n"
			o.write(rec)
			s = 0
			rec = ""


def create_const_number_of_tweets():
	fs = ['500', '600', '700','800']
	for s in fs:
		lines = open("tweets_big_"+s, 'r').readlines()
		deficit = 1000000 - len(lines)
		more_lines = lines[:deficit+1]
		lines.extend(more_lines)
		fo = open("tweets_big_"+s, 'w')
		for l in lines:
			fo.write(l)
		fo.close()


generate_records_of_length(100, 'tweets_big')
generate_records_of_length(200, 'tweets_big')
generate_records_of_length(300, 'tweets_big')
generate_records_of_length(400, 'tweets_big')
generate_records_of_length(500, 'tweets_big')
generate_records_of_length(600, 'tweets_big')
generate_records_of_length(700, 'tweets_big')
generate_records_of_length(800, 'tweets_big')