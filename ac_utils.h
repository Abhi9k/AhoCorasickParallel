#ifndef ACUTILS_H__
#define ACUTILS_H__
#include "utils.h"
#include <queue>

using namespace std;

const int NUM_ROWS = 6000;
const int NUM_COLS = 30;

void generate_DFA(int* dfa, string output[], vector<string> input);
void generate_fail_states(int* dfa, int output[], int fail_state[]);
int get_state_as_int(char ch);
char get_state_as_char(int st);


#endif

