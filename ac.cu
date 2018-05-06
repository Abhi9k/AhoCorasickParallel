#include "ac.h"

__global__
void
profanity_filter_cuda(int* dfa, int* fail_state, unsigned char* tweets, bool* valid_state, int offset, int num_tweets) {
	// int blockPos = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;
	// int threadPos = blockPos*(gridDim.x*gridDim.y*gridDim.z) + (threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x);

	int threadPos = blockIdx.x * blockDim.x + threadIdx.x;
	//threadPos = threadIdx.x;
	// for(int i=0; i< 900; i++)
	// 	printf("%c",tweets[threadPos*tweets_pitch +i]);
	// __syncthreads();

	//unsigned char* tweet = (unsigned char *)((unsigned char *)tweets + threadPos*tweets_pitch);
	//unsigned char* tweet = (unsigned char*)tweets+(threadPos*max_tweet_length);

	int t_row = threadPos*max_tweet_length;
	int total_size = num_tweets*max_tweet_length;


	int curr_state = 0;
	int idx = 0;
	unsigned char ch;
	while((t_row + idx < total_size-1) && ((ch = tweets[t_row + idx++])!=10)) {
		
		int ord;
		//ord = get_state_as_int(chx);
		//cannot call above function as this will be in host
		//cuda does not allow calling host function from gpu
		//thus we will write the logic of above function here
		//even though its code duplication
		//will find better solution later
		if(ch == ' ')
			ord = 28;
		else if(int(ch) == 39)
			ord = 29;
		else
			ord = int(ch) - int('a') + 1;

		if(ord <0 && ord >=30)
			continue;
		printf("%c, %d, %d\n", ch, curr_state, ord);
		int d_row = curr_state*NUM_COLS;

		// //int* curr_state_dfa = (int *)dfa+(curr_state*NUM_COLS);

		while(curr_state!=0 && dfa[d_row + ord] == 0)
			curr_state = fail_state[curr_state];
		if(curr_state==0 && dfa[d_row + ord]==0)
			continue;
		else if(d_row < NUM_ROWS) {
			curr_state = dfa[d_row + ord];
			if(dfa[d_row + 0] == 1) {
				valid_state[offset + threadPos] = true;
				printf("%d,", offset + threadPos);
				break;
			}
		}
	}

}

void
profanity_filter_parallel(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets) {
	int num_blocks = 1;
	int num_threads = 1;
	int* d_dfa;
	int* d_fail_state;
	unsigned char* d_tweets;
	bool* d_valid_state;

	cout<<"inside parallel"<<endl;

	cudaMalloc((void **)&d_fail_state, NUM_ROWS*sizeof(int));
	cudaMalloc((void **)&d_valid_state, num_tweets*sizeof(bool));
	cudaMalloc((void **)&d_dfa, NUM_COLS*NUM_ROWS*sizeof(int));
	
	//cudaMallocPitch((void **)&d_tweets, &tweets_pitch, max_tweet_length*sizeof(unsigned char), num_blocks*num_threads);
	//cudaMemcpy2D(d_dfa, dfa_pitch, dfa, NUM_COLS*sizeof(int), NUM_COLS*sizeof(int), NUM_ROWS, cudaMemcpyHostToDevice);

	// cudaMemcpy2D(d_tweets, tweets_pitch, tweets, max_tweet_length*sizeof(char), max_tweet_length*sizeof(char), num_tweets, cudaMemcpyHostToDevice);



	cudaMemcpy(d_fail_state, fail_state, NUM_ROWS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dfa, dfa, NUM_ROWS*NUM_COLS*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(d_valid_state, false, num_tweets*sizeof(bool));


	for(int i=0; i<num_tweets; i+=(num_blocks*num_threads)) {

	int chunk_size;
	if(i+num_blocks*num_threads >num_tweets)
		chunk_size = num_tweets - i - 1;
	else
		chunk_size = num_blocks*num_threads;
	cudaMalloc((void **)&d_tweets, chunk_size*max_tweet_length*sizeof(unsigned char));
	cudaMemcpy(d_tweets, (tweets+i*max_tweet_length), chunk_size*max_tweet_length*sizeof(unsigned char), cudaMemcpyHostToDevice);
	//cudaMemcpy2D(d_tweets, tweets_pitch, tweets+(i*max_tweet_length*sizeof(unsigned char)), max_tweet_length*sizeof(unsigned char), max_tweet_length*sizeof(unsigned char), num_blocks*num_threads, cudaMemcpyHostToDevice);

		profanity_filter_cuda<<<dim3(num_blocks,1,1), dim3(num_threads,1,1)>>>(d_dfa, d_fail_state, d_tweets, d_valid_state, i, num_tweets);
		cudaDeviceSynchronize();

	}

	

	cudaMemcpy(valid_state, d_valid_state, num_tweets*sizeof(bool), cudaMemcpyDeviceToHost);

	// for(int i=0; i< num_tweets; i++) {
	// 	if(valid_state[i] == false)
	// 		cout<<i<<",";
	// }
	cudaFree(d_dfa);
	cudaFree(d_fail_state);
	cudaFree(d_tweets);
	cudaFree(d_valid_state);
	fflush(stdout);


}