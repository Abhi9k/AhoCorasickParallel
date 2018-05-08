#include "ac.h"
texture<int, cudaTextureType1D> tex_state_final;
texture<int, cudaTextureType1D> tex_dfa;
texture<int, cudaTextureType1D> tex_fail_state;

void
populate_final_states(int* final, int* dfa) {	
	for(int i=0; i<NUM_ROWS; i++) {

		final[i] = dfa[i*NUM_COLS + 0];

	}
}
__global__
void
profanity_filter_cuda(int* dfa, int* fail_state, unsigned char* tweets, bool* valid_state, int offset, int num_tweets, int tweet_length) {
	// for(int i=0; i<TWEET_CHUNK; i++) {

		//int threadPos = blockIdx.x * blockDim.x + threadIdx.x;

		// int t_row = (offset + threadPos*TWEET_CHUNK + i)*max_tweet_length;
		int total_size = num_tweets*tweet_length;

		int num_tweets_per_block = num_tweets/gridDim.x;
		int num_tweets_per_thread = num_tweets/(gridDim.x*blockDim.x);

		int start = blockIdx.x*num_tweets_per_block + threadIdx.x*num_tweets_per_thread;
		int end = start + num_tweets_per_thread;

		int start_ptr = start*tweet_length;
		//int end_ptr = end*max_tweet_length;


		int curr_state = 0;
		int idx = 0;
		int r_idx = 0;
		unsigned char ch;

		while(r_idx < num_tweets_per_thread && (start + r_idx) < num_tweets) {
//printf("processing tweet %d\n", start+r_idx);
			ch = tweets[start_ptr + (r_idx*tweet_length) + idx++];
			//printf("%c", ch);
			if(ch == 10) {
				r_idx += 1;
				curr_state = 0;
				idx = 0;
				continue;
			}
			int ord;
			//ord = get_state_as_int(chx);
			// cannot call above function as this will be in host cuda does not allow calling host function from gpu
			//thus we will write the logic of above function here even though its code duplication
			//will find better solution later

			ord = int(ch) - int('a') + 1;
			if(ch == ' ')
				ord = 28;
			else if(int(ch) == 39)
				ord = 29;

//printf("%d %d\n", ch, curr_state);
			while(curr_state!=0 && tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord) == 0){
				curr_state = tex1Dfetch (tex_fail_state, curr_state);
			}

			if(curr_state!=0 || tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord)!=0) {
				curr_state = tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord);
				int r = tex1Dfetch ( tex_state_final, curr_state );
				if(r) {
					valid_state[start + r_idx] = true;
				}

			}
		}


		// while((t_row + idx < total_size-1) && ((ch = tweets[t_row + idx++])!=10)) {
		// 	int ord;
		// 	//ord = get_state_as_int(chx);
		// 	//cannot call above function as this will be in host
		// 	//cuda does not allow calling host function from gpu
		// 	//thus we will write the logic of above function here
		// 	//even though its code duplication
		// 	//will find better solution later
		// 	if(ch == ' ')
		// 		ord = 28;
		// 	else if(int(ch) == 39)
		// 		ord = 29;
		// 	else
		// 		ord = int(ch) - int('a') + 1;

		// 	if(ord <0 && ord >=30)
		// 		continue;

		// 	while(curr_state!=0 && tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord) == 0){
		// 		curr_state = tex1Dfetch (tex_fail_state, curr_state);
		// 	}

		// 	if(curr_state!=0 || tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord)!=0) {
		// 		curr_state = tex1Dfetch (tex_dfa, curr_state*NUM_COLS + ord);
		// 		int r = tex1Dfetch ( tex_state_final, curr_state );
		// 		if(r) {
		// 			valid_state[offset + threadPos*TWEET_CHUNK + i] = true;
		// 			break;
		// 		}

		// 	}
		// }
	// }

}

void
profanity_filter_parallel(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets, int tweet_length) {
	int num_blocks = 256;
	int num_threads = 1024;

	if(num_tweets < num_blocks*num_threads) {
		num_blocks = 128;
		num_threads = num_tweets/num_blocks;
	}
	int* d_dfa;
	int* d_fail_state;
	unsigned char* d_tweets;
	bool* d_valid_state;
	int* s_final;
	int* final = (int *) malloc(NUM_ROWS*sizeof(int));
	populate_final_states(final, dfa);

	cudaMalloc((void **)&d_fail_state, NUM_ROWS*sizeof(int));
	cudaMalloc((void **)&d_valid_state, num_tweets*sizeof(bool));
	cudaMalloc((void **)&d_dfa, NUM_COLS*NUM_ROWS*sizeof(int));
	cudaMalloc((void **)&s_final, NUM_ROWS*sizeof(int));

	cudaMemcpy(d_fail_state, fail_state, NUM_ROWS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dfa, dfa, NUM_ROWS*NUM_COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(s_final, final, NUM_ROWS*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(d_valid_state, false, num_tweets*sizeof(bool));

	cudaMalloc((void **)&d_tweets, num_tweets*tweet_length*sizeof(unsigned char));
	cudaMemcpy(d_tweets, tweets, num_tweets*tweet_length*sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaBindTexture ( 0, tex_state_final, s_final, NUM_ROWS*sizeof(int) );
	cudaBindTexture ( 0, tex_dfa, d_dfa, NUM_ROWS*NUM_COLS*sizeof(int) );
	cudaBindTexture ( 0, tex_fail_state, d_fail_state, NUM_ROWS*sizeof(int) );
	cudaFuncSetCacheConfig(profanity_filter_cuda, cudaFuncCachePreferL1);
	profanity_filter_cuda<<<dim3(num_blocks,1,1), dim3(num_threads,1,1)>>>(d_dfa, d_fail_state, d_tweets, d_valid_state, 0, num_tweets, tweet_length);
	cudaMemcpy(valid_state, d_valid_state, num_tweets*sizeof(bool), cudaMemcpyDeviceToHost);

	cudaUnbindTexture ( tex_state_final );
	cudaUnbindTexture ( tex_dfa );
	cudaUnbindTexture ( tex_fail_state );
	cudaFree(d_dfa);
	cudaFree(s_final);
	cudaFree(d_fail_state);
	cudaFree(d_tweets);
	cudaFree(d_valid_state);
}