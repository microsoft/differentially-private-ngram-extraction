"""
Extracts Ngrams with Differentially Private Ngrams Extraction (DPNE) method.
Example: run the run.cmd in this folder while in current dir src/python/src
"""
import argparse
import sys
import time
import os
import logging
import math
import numpy as np
import scipy.stats

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType
from pyspark.sql.functions import udf
from shrike.compliant_logging import DataCategory, enable_compliant_logging, prefix_stack_trace

if os.path.exists('dpne.zip'):
    sys.path.insert(0, 'dpne.zip')

from dpne.gaussian_process import GaussianProcess
from dpne.dpne_utils import reservoir_sample_df, log, extract_ngrams_to_validate, extract_ngrams_to_validate_tail, read_with_header, start_spark, write


def get_arg_parser(parser=None):
    """Parse args and do basic checks. 
    
    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        argparse.ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    
    """
    # add arguments that are specific to the module
    if parser is None:
        parser = argparse.ArgumentParser(description='Extracts differentially private ngrams extraction (DPNE) for a dataset with ngrams')

    parser.add_argument(
        '-t',
        '--outputformat',
        required=True,
        help='Output format: json or tsv'
    )
    parser.add_argument(
        '-f',
        '--file_type',
        choices=["csv", "tsv", "json", "parquet"],
        required=True,
        help='One of ["csv", "tsv", "json", "parquet"] for input file type')
    parser.add_argument(
        '-l',
        '--contribution_limit',
        required=False,
        type=int,
        default=1000,
        help='per-user contribution limit'
    )
    parser.add_argument(
        '-pf',
        '--persist_flags',
        required=True,
        type=str,
        default='11',
        help="flag to invoke DataFrame persist on each step or not"
    )
    parser.add_argument(
        '-lf',
        '--log_flags',
        required=True,
        type=str,
        default='11',
        help="flag to log output numbers on each step or not"
    )
    parser.add_argument(
        '-k',
        '--top_k',
        required=True,
        type=int,
        default=1,
        help="flag to select top k frequent n-grams instaed of random sampling from each user"
    )
    parser.add_argument(
        "-du",
        "--delta_user_count",
        required=True,
        type=int,
        default=1,
        help="Use delta = 1/(n*log(n)) where n is number of users when value is 1, otherwise use constant value"
    )
    parser.add_argument(
        '-n',
        '--ngram_size',
        required=False,
        type=int,
        default=1,
        help='Max ngram length to extract (will extract 1 to args.ngram_size)'
    )
    parser.add_argument(
        '-s',
        '--estimate_sample_size',
        required=False,
        type=float,
        default=0.01,
        help='subsampling ratio to estimate the valid n-grams'
    )
    parser.add_argument(
        '-b',
        '--budget_distribute',
        required=False,
        type=float,
        default=1.0,
        help='the base of exponential term for distributing budget, if 1.0, it is distributing uniformly'
    )
    parser.add_argument(
        '--filter_one_side',
        required=False,
        type=int,
        default=0,
        help='Filter only one side as we did for the bugged version. This is for comparison, use value 1 for default experience'
    )
    parser.add_argument(
        '-e',
        '--dp_epsilon',
        required=False,
        type=float,
        default=3.0,
        help='Epsilon for differential privacy'
    )
    parser.add_argument(
        '-et',
        '--dp_eta',
        required=False,
        type=float,
        default=0.01,
        help='Max fraction of spurious ngrams that we can tolerate'
    )
    parser.add_argument(
        "-d",
        "--dp_delta",
        required=True,
        type=float,
        default=0.0,
        help="constant value to use as delta. If value is 0, use e^(-10)"
    )
    parser.add_argument(
        '-i',
        '--input_path', # inputs MUST begin with --input_
        required=True,
        type=str,
        help='path/filename to the input file')
    parser.add_argument(
        '-o',
        '--output_path', # outputs MUST begin with --output_
        required=True,
        type=str,
        help='path/filename to the output file to write the output (json format)')
    return parser


def compute_dpne(cur_ngram_size, input_ngrams, list_persist_flags, list_log_flags, args, prev_ngrams_extracted=None, vocab=None, num_users_in_data=0, vocab_count=0, prev_ngrams_count=0):
    """
    Compute the differentially-private set union of the input dataframe
    """
    epsilon = args.dp_epsilon
    delta = math.exp(-10) if args.dp_delta == 0.0 else args.dp_delta
    eta = args.dp_eta
    # make sure contribution limit is at least 1
    tokens_per_user = args.contribution_limit if args.contribution_limit > 0 else 1
    subsampling_ratio = args.estimate_sample_size if (args.estimate_sample_size > 0 and args.estimate_sample_size < 1.0) else 1.0
    filter_one_side = True if args.filter_one_side == 1 else False
    budget_distribute = float(args.budget_distribute)

    sample_top_k = True if args.top_k == 1 else False
    if sample_top_k:
        log(logging.INFO, DataCategory.PUBLIC, "Sampling top k n-grams instead of random sampling from each user..")

    delta_user_count = True if args.delta_user_count == 1 else False
    extract_ngrams_to_validate_udf = udf(extract_ngrams_to_validate, StringType())
    extract_ngrams_to_validate_tail_udf = udf(extract_ngrams_to_validate_tail, StringType())
    num_users = num_users_in_data

    log(logging.INFO, DataCategory.PUBLIC, "total {}-grams: {}".format(cur_ngram_size, input_ngrams.count()))

    if cur_ngram_size > 1:
        if delta_user_count:
            # use delta = 1/(N*log(N)) if delta_user_count flag is set
            delta = 1.0 / (num_users * math.log(num_users))

        # randomly sample (with the ratio given) previous valid n-grams and token from vocab
        # according to https://stackoverflow.com/questions/39344769/spark-dataframe-select-n-random-rows, this way gives more accurate # of samples
        sqrt_subsampling_ratio = math.sqrt(subsampling_ratio)
        num_vocab_sampling = int(max(sqrt_subsampling_ratio * vocab_count, 1))
        num_prev_ngrams_sampling = int(max(sqrt_subsampling_ratio * prev_ngrams_count, 1))
        vocab_sampled = vocab.orderBy(F.rand()).limit(num_vocab_sampling).alias("vocab_sampled")
        prev_ngrams_extracted_sampled = prev_ngrams_extracted.orderBy(F.rand()).limit(num_prev_ngrams_sampling).alias("prev_ngrams_sampled")
        #prev_ngrams_extracted_sampled = prev_ngrams_extracted.sample(sqrt_subsampling_ratio).alias("prev_ngrams_sampled")
        #vocab_sampled = vocab.sample(sqrt_subsampling_ratio).alias("vocab_sampled")

        # this will generate n-grams with previous (n-1)-grams + " " + one of the vocab token
        ngrams_sample_to_validate = prev_ngrams_extracted_sampled.crossJoin(vocab_sampled.select(F.col("ngrams"))).select(F.concat(F.col("prev_ngrams_sampled.ngrams"), F.lit(" "), F.col("vocab_sampled.ngrams")).alias("ngrams")).alias("cur_ngrams_sampled")
        ngrams_sample_to_validate_count = ngrams_sample_to_validate.count()
        log(logging.INFO, DataCategory.PUBLIC, "n-grams to check: {}".format(ngrams_sample_to_validate_count))

        # this will extract the ngram[1:] portion to use for validation
        ngrams_sample_to_check = ngrams_sample_to_validate.withColumn("ngrams_to_check", extract_ngrams_to_validate_udf(F.col("ngrams")))

        # get the current valid n-grams from sampled n-grams
        valid_ngrams_samples = ngrams_sample_to_check.join(prev_ngrams_extracted, ngrams_sample_to_check.ngrams_to_check == prev_ngrams_extracted.ngrams, "left_semi").select(F.col("ngrams")).alias("valid_ngrams_samples")
        valid_samples_count = valid_ngrams_samples.count()
        # calculate estimate valid ngrams in the current step
        estimated_num_valid_ngrams = int(1 + (valid_samples_count/ngrams_sample_to_validate_count) * prev_ngrams_count * vocab_count)
        log(logging.INFO, DataCategory.PUBLIC, "estimated valid # of n-grams: {}".format(estimated_num_valid_ngrams))

        dpsu = GaussianProcess(cur_ngram_size, epsilon, delta, eta, tokens_per_user, args.ngram_size, budget_distribute, estimated_num_valid_ngrams, prev_ngrams_count)

        # find all distinct n-grams from the partial input
        distinct_input_ngrams = input_ngrams.select(F.col("ngrams")).distinct().alias("distinct_ngrams")
        log(logging.INFO, DataCategory.PUBLIC, "dstinct # of n-grams: {}".format(distinct_input_ngrams.count()))

        # extract the portion (ngram[1:]) to validate from whole n-grams we have in this step
        distinct_input_ngrams_to_check = distinct_input_ngrams.withColumn("ngrams_to_check", extract_ngrams_to_validate_udf(F.col("ngrams")))
        # get only valid n-grams from the whole n-grams from input data
        valid_distinct_input_ngrams = distinct_input_ngrams_to_check.join(prev_ngrams_extracted, distinct_input_ngrams_to_check.ngrams_to_check == prev_ngrams_extracted.ngrams, "left_semi").select(F.col("ngrams"))

        # also extract the portion (ngram[:-1]) to valiate
        valid_distinct_input_ngrams = valid_distinct_input_ngrams.withColumn("ngrams_to_check_tail", extract_ngrams_to_validate_tail_udf(F.col("ngrams")))

        if not filter_one_side:
            valid_distinct_input_ngrams = valid_distinct_input_ngrams.join(prev_ngrams_extracted, valid_distinct_input_ngrams.ngrams_to_check_tail == prev_ngrams_extracted.ngrams, "left_semi").select(F.col("ngrams"))
        
        else:
            # if we only filter one side, try to have a separate table that has filtered with both side
            valid_distinct_input_ngrams_both = valid_distinct_input_ngrams.join(prev_ngrams_extracted, valid_distinct_input_ngrams.ngrams_to_check_tail == prev_ngrams_extracted.ngrams, "left_semi").select(F.col("ngrams"))

        log(logging.INFO, DataCategory.PUBLIC, "actual valid # of n-grams: {}".format(valid_distinct_input_ngrams.count()))

        # get only valid input ngrams
        cur_input_ngrams = input_ngrams.join(valid_distinct_input_ngrams, "ngrams").select("partial_input.*").filter(F.col("ngrams").isNotNull())
        # sample from only valid input n-grams in this step
        sampled = reservoir_sample_df(cur_input_ngrams, tokens_per_user, topk=sample_top_k)

        # sum up all counts for same n-gram
        counted = sampled.groupBy("ngrams").agg(F.sum("count"), F.sum("raw_count"))
        counted = counted.select(F.col("ngrams").alias("ngrams"), F.col("sum(count)").alias("count"), F.col("sum(raw_count)").alias("raw_count")).alias("conted_ngrams")

        if filter_one_side:
            # see how many of those n-grams selected were from the n-grams filtered with both side
            ngrams_not_filtered_in_both_side = counted.join(valid_distinct_input_ngrams_both, counted.ngrams == valid_distinct_input_ngrams_both.ngrams)
            log(logging.INFO, DataCategory.PUBLIC, "extracted n-grams survived with both side filtering: {}".format(ngrams_not_filtered_in_both_side.count()))

    else:
        # this is just for logging purpose, to get the distinct # of unigrams.
        distinct_input_ngrams = input_ngrams.select(F.col("ngrams")).distinct().alias("distinct_ngrams")
        log(logging.INFO, DataCategory.PUBLIC, "dstinct # of n-grams: {}".format(distinct_input_ngrams.count()))

        # we can use all unigram input as valid ngrams. Simply run DPSU on unigrams.
        sampled = reservoir_sample_df(input_ngrams, tokens_per_user, topk=sample_top_k)

        # find total number of users
        if delta_user_count:
            num_users = int(sampled.select("user").distinct().count())
            log(logging.INFO, DataCategory.PUBLIC, "use delta=1/(n*log(n)) equation, num users = {}".format(num_users))
            delta = 1.0 / (num_users * math.log(num_users))

        dpsu = GaussianProcess(cur_ngram_size, epsilon, delta, eta, tokens_per_user, args.ngram_size, budget_distribute)

        # sum up all counts for same n-gram
        counted = sampled.groupBy("ngrams").agg(F.sum("count"), F.sum("raw_count"))
        counted = counted.select(F.col("ngrams").alias("ngrams"), F.col("sum(count)").alias("count"), F.col("sum(raw_count)").alias("raw_count"))

    # Extract n-grams above threshold after adding noise
    exceeds_threshold_udf = udf(dpsu.exceeds_threshold, BooleanType())
    cur_ngrams_extracted = counted.filter(exceeds_threshold_udf("count"))
    # test to see if this helps to solve the invalid udf problem after join method.. 
    if list_persist_flags[1]:
        cur_ngrams_extracted.persist()

    # add suprious n-grams
    if cur_ngram_size > 1:
        cur_ngrams_extracted_count = cur_ngrams_extracted.count()
        num_spurious_ngrams = np.random.binomial(max(0, estimated_num_valid_ngrams - cur_ngrams_extracted_count), scipy.stats.norm.cdf(-dpsu.g_rho / dpsu.g_param))
        log(logging.INFO, DataCategory.PUBLIC, "adding spurious n-grams: {}".format(num_spurious_ngrams))

        spurious_counter = 0

        while spurious_counter < num_spurious_ngrams:
            # sample 2*sampling ratio to find enough spurious ngrams
            sqrt_eta = math.sqrt(eta*2)
            num_vocab_sampling = int(max(sqrt_eta * vocab_count, 1))
            num_prev_ngrams_sampling = int(max(sqrt_eta * prev_ngrams_count, 1))
            vocab_sampled = vocab.orderBy(F.rand()).limit(num_vocab_sampling).alias("vocab_sampled")
            prev_ngrams_extracted_sampled = prev_ngrams_extracted.orderBy(F.rand()).limit(num_prev_ngrams_sampling).alias("prev_ngrams_sampled")

            # prev_ngrams_extracted_sampled = prev_ngrams_extracted.sample(True, sqrt_eta).alias("prev_ngrams_sampled")
            # vocab_sampled = vocab.sample(sqrt_eta).alias("vocab_sampled")

            spurious_ngram_candidate = prev_ngrams_extracted_sampled.crossJoin(vocab_sampled.select(F.col("ngrams"))).select(F.concat(F.col("prev_ngrams_sampled.ngrams"), F.lit(" "), F.col("vocab_sampled.ngrams")).alias("ngrams"), F.lit(0.0).alias("count"), F.lit(0).alias("raw_count")).alias("spurious_ngram_candidate")
            spurious_ngram_candidate = spurious_ngram_candidate.filter(F.col("ngrams").isNotNull())

            # now we have to check if they are already in the list or not
            spurious_ngrams = spurious_ngram_candidate.join(cur_ngrams_extracted, "ngrams", "left_anti").select("spurious_ngram_candidate.*").filter(F.col("ngrams").isNotNull())

            # now check the count
            current_sampled_spurious = spurious_ngrams.count()
            if spurious_counter + current_sampled_spurious < num_spurious_ngrams:
                # we have to add all
                spurious_counter += current_sampled_spurious
                cur_ngrams_extracted = cur_ngrams_extracted.union(spurious_ngrams)
            else:
                # we need to take top reamining required spurious ngrams
                spurious_ngrams = spurious_ngrams.orderBy(F.rand()).limit(num_spurious_ngrams - spurious_counter)
                #ratio_to_subsample = (num_spurious_ngrams - spurious_counter) / float(current_sampled_spurious)
                #spurious_ngrams = spurious_ngrams.sample(True, ratio_to_subsample)
                cur_ngrams_extracted = cur_ngrams_extracted.union(spurious_ngrams)
                # since we are done, break the while loop
                break

    results = cur_ngrams_extracted if cur_ngrams_extracted.count() > 0 else None
    return results, num_users


@prefix_stack_trace()
def execute(args):
    """
    Compute differentially private n-grams extraction.
    """
    spark = start_spark(sys.argv[0])
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # Uncomment the following when running locally to boost up local speed
    #spark.conf.set('spark.sql.shuffle.partitions', 2)

    persist_flags = args.persist_flags
    log_flags = args.log_flags

    # if the flags has wrong format, just add everything to True. log flags has one less step since it is not printing out after tokenization.
    if len(persist_flags) < 2:
        persist_flags = '11'
    if len(log_flags) < 2:
        log_flags = '11'
    
    # parse the flags
    list_persist_flags = [True if x == '1' else False for x in persist_flags]
    list_log_flags = [True if x == '1' else False for x in log_flags]

    # read unigrams from input
    input_unigram_path = os.path.join(args.input_path, "1gram")
    unigrams = read_with_header(input_unigram_path, args.file_type, spark)
    vocab, num_users = compute_dpne(1, unigrams, list_persist_flags, list_log_flags, args)

    if list_persist_flags[0]:
        vocab.persist()

    if vocab:
        vocab = vocab.alias("vocab")
        vocab_count = vocab.count()
        previous_ngram_count = vocab_count
        total_count = vocab_count
        log(logging.INFO, DataCategory.PUBLIC, "Retrieved 1-gram: {}".format(vocab_count))
        vocab_path = os.path.join(args.output_path, "1gram")
        write(vocab, vocab_path, file_type=args.outputformat, header=True)

        for ngram_size in range(2, args.ngram_size + 1):
            # reads ngram extracted from previous iteration and vocab. 
            # multiple iteration of joins made the PySpark job failure on UDF, to be safe we always read a dataframe from the stored output. 
            prev_ngram_path = os.path.join(args.output_path, "{}gram".format(ngram_size-1))
            prev_ngrams_extracted = read_with_header(prev_ngram_path, args.outputformat, spark)
            prev_ngrams_extracted = prev_ngrams_extracted.alias("prev_ngrams")

            vocab = read_with_header(vocab_path, args.outputformat, spark)
            vocab = vocab.alias("vocab")

            input_ngram_path = os.path.join(args.input_path, "{}gram".format(ngram_size))
            input_ngram = read_with_header(input_ngram_path, args.file_type, spark).alias("partial_input")
            ngrams_extracted, _ = compute_dpne(ngram_size, input_ngram, list_persist_flags, list_log_flags, args, prev_ngrams_extracted, vocab, num_users, vocab_count, previous_ngram_count)
            if list_persist_flags[0]:
                ngrams_extracted.persist()

            if ngrams_extracted:
                extracted_count = ngrams_extracted.count()
                log(logging.INFO, DataCategory.PUBLIC, "Retrieved {}-gram: {}".format(ngram_size, extracted_count))
                total_count += extracted_count
                previous_ngram_count = extracted_count
                ngram_path = os.path.join(args.output_path, "{}gram".format(ngram_size))
                write(ngrams_extracted, ngram_path, file_type=args.outputformat, header=True)
            else:
                log(logging.INFO, DataCategory.PUBLIC, "No {}-grams Retrieved".format(ngram_size))
                break

    log(logging.INFO, DataCategory.PUBLIC, "Retrieved total: {}".format(total_count))

    # Finish up
    log(logging.INFO, DataCategory.PUBLIC, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    spark.stop()


def main(flags=None):
    """ Script main function """
    if not flags:
        flags = sys.argv

    module_args = flags[1:]

    log(logging.INFO, DataCategory.PUBLIC, "Read parameters...")
    parser = get_arg_parser()
    args, _ = parser.parse_known_args(module_args)
    log(logging.INFO, DataCategory.PUBLIC, "Finished reading parameters.")

    execute(args)


if __name__ == '__main__':
    enable_compliant_logging()
    main()
