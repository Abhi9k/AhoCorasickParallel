#include "ac.h"

using namespace std;

void
profanity_filter_serial(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets) {
	for(int i=0; i<num_tweets; i++) {
		char* tweet = (tweets +i*max_tweet_length);

		int curr_state = 0;
		int idx = 0;
		//int tweet_size = strlen(tweet);
		char ch;
		//char ch = tweet[idx++];
		while((ch = tweet[idx++])!=10) {

			int ord = get_state_as_int(ch);

			while(curr_state!=0 && dfa[curr_state*NUM_COLS + ord] == 0)
				curr_state = fail_state[curr_state];
			if(curr_state==0 && dfa[curr_state*NUM_COLS + ord]==0)
				continue;

			curr_state = dfa[curr_state*NUM_COLS + ord];
			if(dfa[curr_state*NUM_COLS + 0] == 1) {
				valid_state[i] = true;
				break;
			}
		}
	}
}