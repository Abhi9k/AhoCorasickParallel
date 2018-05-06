#include "timer.h"
#include "utils.h"
#include "ac.h"


using namespace std;

const int num_cols = 30;
const int num_rows = 6000;
void generate_DFA(int* dfa, string output[], vector<string> input);
void generate_fail_states(int* dfa, string output[], int fail_state[]);
int get_state_as_int(char ch);
char get_state_as_char(int st);


float
ms_difference(struct timeval start, struct timeval end) {
    float ms = (end.tv_sec - start.tv_sec) * 1000;
    ms += (end.tv_usec - start.tv_usec) / 1000;
    return ms;
}

void
print_usage() {
	printf("usage: ./acp [-l filename] [-p] [-t]\n"
		   "              -l load the filename to create dfa\n"
		   "              -p run performance tests\n"
		   "              -t run test cases\n"
		  );
}

std::vector<std::string>
load_input_file(const char* filename, int* n) {
	fstream in;
	// if((in = fopen(filename, "r")) == 0) {
	// 	fprintf(stderr, "Error, could not open file '%s'.\n", filename);
	// 	abort();
	// }
	in.open(filename, fstream::in);

	vector<string> word_list;
	string word;
	int idx = 0;
	while(std::getline(in, word)) {
		word_list.push_back(word);
		idx += 1;
	}
	*n = idx;
	in.close();
	return word_list;
}

char*
load_tweets(const char* filename, int num) {
	FILE * tweetfile;
	tweetfile = fopen(filename, "r");
	char* tweets;
	tweets =  (char *) malloc(num*max_tweet_length*sizeof(char));
	int idx = 0;
	char ch;
	while(idx < num) {
		if(feof(tweetfile))
			break;
		int i = 0;
		do{
			ch = fgetc(tweetfile);
			tweets[idx*max_tweet_length + i] = ch;
			i+=1;
		}while(i<max_tweet_length && ch!=10);

		while(i<max_tweet_length) {
			tweets[idx*max_tweet_length + i] = ' ';
			i+= 1;
		}
		idx += 1;
	}
	fclose(tweetfile);

	return tweets;
}

void
print_dfa(int* dfa, int rows, int cols) {
	const char* format_int = "%4d|";
	const char* format_char = "%4c|";
	const char* format_str = "%4s|";
	printf(format_str, " ");
	printf(format_char, '0');
	for(int i=1; i<cols; i++) {
		printf(format_char, get_state_as_char(i));
	}
	cout<<endl;
	for(int i=0; i<cols+1; i++) {
		printf("%4s", "-----");
	}
	cout<<endl;
	for(int i=0; i<rows; i++) {
		printf(format_int, i);
		printf(format_int, dfa[i*num_cols + 0]);
		for(int j=1; j<cols; j++) {
			if(dfa[i*num_cols + j] != 0)
				printf("%4d|", dfa[i*num_cols + j]);
			else
				printf(format_str, ".");
		}
		cout<<endl;
	}
}

void
print_word_list(vector<string> word_list, int num, int total_words) {
	if(num > total_words){
		num = total_words;
	}
	cout<<"total number of words: "<<total_words<<endl;
	cout<<"printing words: "<<num<<endl;

	int start = 0;

	for(vector<string>::iterator it=word_list.begin(); start < num; ++it, start++) {
		cout<<*it<<endl;
	}
}

void
print_state_outputs(int* dfa, string output[], int num_rows) {
	for(int i=0; i<num_rows; i++) {
		if(dfa[i*num_cols + 0]!=0) {
			cout<<"state:"<<i<<" "<<output[i]<<endl;
		}
	}
}

void
print_fail_states(int fail_state[], int num_states) {
	for(int i=0; i<num_states; i++) {
		cout<<i<<" "<<fail_state[i]<<endl;
	}
}

void
performance_test(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets) {
	cout<<"running performance tests"<<endl;
	// serial performance
	struct timeval start, end;
	float serial_time, parallel_time;
	fflush(stdout);
	gettimeofday(&start, NULL);
	profanity_filter_serial(dfa, fail_state, tweets, valid_state, num_tweets);
	gettimeofday(&end, NULL);
	serial_time = ms_difference(start, end);
	// for(int i=0; i< num_tweets; i++) {
	// 	cout<<valid_state[i]<<",";
	// }
	printf("Your serial code ran in: %f msecs.\n", serial_time);
	// memset(valid_state, false, sizeof(valid_state));
	// fflush(stdout);
	// gettimeofday(&start, NULL);
	// profanity_filter_parallel(dfa, fail_state, tweets, valid_state, num_tweets);
	// gettimeofday(&end, NULL);
	// parallel_time = ms_difference(start, end);
	// for(int i=0; i< 100; i++) {
	// 	cout<<valid_state[i]<<",";
	// }
	// printf("Your parallel code ran in: %f msecs.\n", parallel_time);

}

int main(int argc, char **argv) {

	bool lflag=false, tflag=false, pflag=false;
	int dfa[num_rows*num_cols];
	int num_of_words;
	int c;
	int fail_state[num_rows];
	string output[num_rows];
	char* input_filename;
	vector<string> word_list;
	GpuTimer timer;

	memset(dfa, 0, sizeof(dfa[0]) * num_rows * num_cols);
	memset(fail_state, 0, sizeof(fail_state));

	while((c = getopt(argc, argv, "l:tp?")) != -1) {
		switch(c) {
			case 'l':
				lflag = true;
				input_filename = optarg;
				break;

			case 't':
				tflag = true;
				break;

			case 'p':
				pflag = true;
				break;

			case '?':
				print_usage();
				exit(0);
				break;

			default:
				exit(0);
				break;
		}
	}

	if(!lflag) {
		cout<<"Please provide list of words to load"<<endl;
		exit(0);
	}


	word_list = load_input_file(input_filename, &num_of_words);

	//print_word_list(word_list, 2, num_of_words);

	generate_DFA(dfa, output, word_list);

	//print_dfa(dfa, 100, num_cols);

	generate_fail_states(dfa, output, fail_state);

	//print_state_outputs(dfa, output, num_rows);

	//print_fail_states(fail_state, num_cols);

	const int num_tweets = 100000;
	bool valid_state[num_tweets];

	char* tweets = load_tweets("data/tweets_small", num_tweets);
	// for(int i=0; i<num_tweets; i++){
	// 	cout<<tweets[i]<<endl;
	// }

	memset(valid_state, false, sizeof(valid_state));

	performance_test(dfa, fail_state, tweets, valid_state, num_tweets);


	return 1;
}