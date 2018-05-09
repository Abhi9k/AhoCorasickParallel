# AhoCorasickParallel
Parallel implementation of Aho Corasick algorithm using OpenACC and CUDA

# Test Environment
====================
All the experiments were conducted on ironclaw1 and ironclaw2 machines

	Hardware Specifications
		1. Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
		2. 32 GB memory
		3. 2 Grid K1 GPU
		4. 2 NUMA nodes

	Software Specifications
		1. Operating System: CentOS Linux 7 (Core)
		2. gcc version 4.8.5
		3. NVIDIA (R) Cuda compiler driver V8.0.61

# Code walkthrough
===================
Our entire code is hosted on GitHub at: https://github.com/Abhi9k/AhoCorasickParallel

1. main.cpp: driver file where the execution starts.
			 It is responsible for loading tweets, loading bad words, creating DFA for STT, fail state
			 and running various experiment setups.
2. ac_utils.cpp: contains the logic of creating DFA for STT and fail state table.
3. ac_serial.cpp: contains the serial implementation of AC algorithm
4. ac_open_acc.cpp: contains the OpenACC implementation of AC algorithm
5. ac.cu: contains the CUDA implementation of AC algorithm

'data' directory holds all the input data sets.

# Directions for runnning the experiment
=========================================
The project has a Makefile which takes care of compiling the project.
Please enter the project folder and then follow the steps given below:

	Step 1: make clean  -- to make sure old object files are removed
	Step 2: make
	Step 3: ./acp -l "data/bad_words"

	To run a search to find the best 'block size' and 'thread size' for CUDA run
	following command after Step 1 and 2:

	./acp -l "data/bad_words" -s

/******************************************************************************
After Step 3, you should see something like this on the console:

Type,# of records,# of characters in each record,# of patterns,Runtime (in ms)
SERIAL,100000,100,200,99.149376
CUDA,100000,100,200,31.413729
OpenACC,100000,100,200,54.380608
SERIAL,100000,100,400,54.830593
CUDA,100000,100,400,34.388512
OpenACC,100000,100,400,54.935295
*******************************************************************************/

