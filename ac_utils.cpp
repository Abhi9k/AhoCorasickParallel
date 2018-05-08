#include "ac_utils.h"

using namespace std;

int 
get_state_as_int(char ch) {
	if(ch == ' ')
		return 28;
	if(int(ch) == 39)
		return 29;
	return int(ch) - int('a') + 1;
}

char
get_state_as_char(int st) {
	if(st==28)
		return 32;
	if(st == 29)
		return 39;
	return st-1 + int('a');
}

void
generate_DFA(int* dfa, string output[], vector<string> input) {

	// this variable hold the current state represented as integer
	// integer for next state is determined by increasing it by 1
	int next_state = 0;
	for(vector<string>::iterator it=input.begin(); it != input.end(); ++it) {
		string str = *it;
		int curr_state = 0;
		for(char& ch : str) {
			int st = get_state_as_int(ch);
			if(st < 0 || st>=30)
				continue;
			if(dfa[curr_state*NUM_COLS + st] != 0) {
				curr_state = dfa[curr_state*NUM_COLS + st];
			}
			else {
				next_state += 1;
				dfa[curr_state*NUM_COLS + st] = next_state;
				curr_state = next_state;
			}
		}
		output[curr_state] = str;
		dfa[curr_state*NUM_COLS + 0] = 1;
	}
}

void
generate_fail_states(int* dfa, string output[], int fail_state[]) {
	// hold the list of states for which failure state is yet to be determined
	queue <int> s_queue;

	/**
		Initialize s_queue by all the states at depth 1
		Depth of a state 's' is defined by the length of the
		smallest path from state 0 to state 's'
	**/
	for(int i = 1; i < NUM_COLS; i++) {
		if(dfa[i] == 0)
			continue;
		s_queue.push(dfa[i]);
	}

	while(!s_queue.empty()) {

		int r = s_queue.front();

		// make sure to remove the front element or else ...
		s_queue.pop();

		for(int i = 1; i < NUM_COLS; i++) {
			if(dfa[r*NUM_COLS + i] == 0)
				continue;

			int st = fail_state[r];
			int a = dfa[r*NUM_COLS + i];
			s_queue.push(a);

			if(st != 0) {
				while(dfa[st*NUM_COLS + i] == 0 && st!=0){
					st = fail_state[st];
				}
			}

			fail_state[a] = dfa[st*NUM_COLS + i];


			// TODO: append to output
		}
	}
}