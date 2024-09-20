# This script runs a series of tokenization, ngram extraction, and DP N-gram extraction using the parameters specified as arguments
# first argument = location of source file to extract DPNE from
# second argument = output folder high level
# third argument = file extension - json or anything else.
# fourth argument = epsilon for DP
# fifth argument = the highest N in n-grams to DP-extract

spark-submit dpne/tokenize_text.py -f json --ngrams $5 --max_num_tokens 400 --allow_multiple_ngrams 1 -i $1 -o ./$2/tokenize_text -t $3

spark-submit split_ngrams.py --ngram_size $5 -i ./$2/tokenize_text -o ./$2/split_ngrams -f $3 -t $3

# modified the hyperparams a bit
spark-submit extract_dpne.py --dp_epsilon $4 --dp_eta 0.1 --dp_delta 0.5 --contribution_limit 10 --persist_flags 11 --log_flags 00 --top_k 1 --delta_user_count 0 --ngram_size $5 --filter_one_side 0 --budget_distribute 1.0 --estimate_sample_size 0.8 -i ./$2/split_ngrams -o ./$2/dpne_sample -f $3 -t $3

