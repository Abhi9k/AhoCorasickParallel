#ifndef AC_H__
#define AC_H__
#include "ac_utils.h"
#include <stdio.h>

void profanity_filter_serial(int* dfa, int fail_state[], char* tweets, bool* valid_state, int num_tweets, int tweet_length);
__global__ void profanity_filter_cuda(int* dfa, int* fail_state, unsigned char* tweets, bool* valid_state, int offset, int num_tweets, int tweet_length);

void profanity_filter_parallel(int* dfa, int fail_state[], char* tweets, bool* valid_state, int num_tweets);
void profanity_filter_acc_parallel(int* dfa, int fail_state[], char* tweets, bool* valid_state, int num_tweets);

#endif