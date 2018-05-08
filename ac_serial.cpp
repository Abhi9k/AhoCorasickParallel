#include "ac.h"

using namespace std;

void
profanity_filter_serial(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets, int tweet_length) {
	for(int i=0; i<num_tweets; i++) {
		char* tweet = (tweets +i*tweet_length);

		int curr_state = 0;
		int idx = 0;
		char ch;
		while((ch = tweet[idx++])!=10) {
			int ord = get_state_as_int(ch);
			if(ord < 0 || ord > 29)
				continue;

			while(curr_state!=0 && dfa[curr_state*NUM_COLS + ord] == 0)
				curr_state = fail_state[curr_state];
			if(curr_state==0 && dfa[curr_state*NUM_COLS + ord]==0)
				continue;

			curr_state = dfa[curr_state*NUM_COLS + ord];
			if(dfa[curr_state*NUM_COLS + 0] == 1) {
				int state = curr_state;
				valid_state[i] = true;
				break;
			}
		}
	}
}