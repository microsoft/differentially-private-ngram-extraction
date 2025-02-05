set DATA_HOME=.\sample_data

rem download wildchat 50k reproducible sample
python convert_wildchat.py --output_path %DATA_HOME%

rem rephrase the messages using AzureOpenAI models do provide your endpoint and deployed model name
python ..\rephrase\get_rephrased_messages.py --dataset_path  %DATA_HOME%\ --column message --id message_id --output_path %DATA_HOME%\50k_sample_rephrased\ --max_workers 50 --endpoint "" --model ""

rem tokenize the rephrased messages
spark-submit dpne\tokenize_text.py -f parquet --ngrams 10 --allow_multiple_ngrams 1 -i %DATA_HOME%\50k_sample_rephrased\ -o %DATA_HOME%\tokenize_text -t json

rem split ngrams
spark-submit dpne\split_ngrams.py --ngram_size 10 -i %DATA_HOME%\tokenize_text -o %DATA_HOME%\split_ngrams -f json -t json

rem run DPNE
spark-submit dpne\extract_dpne.py --dp_epsilon 3 --dp_eta 0.01 --dp_delta 0 --contribution_limit 1000 --persist_flags 11 --log_flags 00 --top_k 1 --delta_user_count 1 --ngram_size 10 --filter_one_side 0 --budget_distribute 10.0 --estimate_sample_size 0.01 -i %DATA_HOME%\split_ngrams -o %DATA_HOME%\dpne_extraction -f json -t tsv