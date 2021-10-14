set DATA_HOME=.\sample_data

rem tokenize_emails sample
spark-submit dpne\tokenize_text.py -f json --ngrams 3 --max_num_tokens 400 --allow_multiple_ngrams 1 -i %DATA_HOME%\msnbc_sample.json -o ./output/tokenize_text -t json

rem split_ngrams sample
spark-submit dpne\split_ngrams.py --ngram_size 3 -i ./output/tokenize_text -o ./output/split_ngrams -f json -t json

rem run DPNE
spark-submit dpne\extract_dpne.py --dp_epsilon 100.0 --dp_eta 0.1 --dp_delta 0.5 --contribution_limit 100 --persist_flags 11 --log_flags 00 --top_k 1 --delta_user_count 0 --ngram_size 3 --filter_one_side 0 --budget_distribute 10.0 --estimate_sample_size 0.8 -i ./output/split_ngrams -o ./output/dpne_sample -f json -t json

rem calculate k-anon n-gram coverage
spark-submit dpne\k_anon_coverage.py --need_split 0  --k_values "1,2" -i ./output/split_ngrams --input_ngram_path output/dpne_sample --ngram_size 2 -f json