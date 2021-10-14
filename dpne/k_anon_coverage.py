"""
Get k anonymized ngram and find coverage of extracted ngrams
"""
import argparse
import sys
import time
import os.path
import argparse
import logging

from pyspark.sql import functions as F
from shrike.compliant_logging import DataCategory, enable_compliant_logging, prefix_stack_trace

if os.path.exists('dpne.zip'):
    sys.path.insert(0, 'dpne.zip')
    
from dpne.dpne_utils import log, start_spark, read_with_header


def get_arg_parser(parser=None):
    """Parse args and do basic checks. 
    
    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser): an argument parser instance

    Returns:
        argparse.ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    
    """
    # add arguments that are specific to the module
    if parser is None:
        parser = argparse.ArgumentParser(description='checks the coverage of DPNE n-grams compared with k-anonymized n-grams')

    parser.add_argument(
        '-f',
        '--file_type',
        choices=["csv", "tsv", "json", "parquet"],
        required=True,
        help='One of ["csv", "tsv", "json", "parquet"] for input file type')
    
    parser.add_argument(
        '--k_values',
        required=True,
        type=str,
        help="k values to consider for k-anonymization, separated with comma"
    )

    parser.add_argument(
        '--need_split',
        required=True,
        type=int,
        help="if 1, split the extracted dp n-grams by n-gram size"
    )

    parser.add_argument(
        '--ngram_size',
        required=True,
        type=int,
        help="max ngram size to consider"
    )
    
    parser.add_argument(
        '-i',
        '--input_path',
        required=True,
        type=str,
        help='path/filename to the input file')
    
    parser.add_argument(
        '--input_ngram_path', 
        required=True,
        type=str,
        help='path/filename to the extracted dp ngram file')

    return parser


def safe_count(rdd):
    return 0 if rdd.isEmpty() else rdd.count()


def compute_k_anon(input_df, ngram_df, args):
    """
    Get k anonymized ngram and find coverage of extracted ngrams
    """
    k_values_str = args.k_values
    k_values_arr = [int(i) for i in k_values_str.strip().split(",")]

    for k_value in k_values_arr:
        distinct_input_ngrams = input_df.distinct().alias("distinct_ngrams")
        distinct_ngrams_count = distinct_input_ngrams.groupBy(F.col("ngrams")).agg({"*":"count"})
        k_anon_ngrams = distinct_ngrams_count.filter(F.col("count(1)") > k_value)
        
        ngram_df_covered = ngram_df.join(k_anon_ngrams, ngram_df.ngrams == k_anon_ngrams.ngrams)
        ngram_df_covered.persist()

        log(logging.INFO, DataCategory.PUBLIC, "k = {} : k-anon total={}, extracted total={}, covered total={}".format(k_value, k_anon_ngrams.count(), ngram_df.count(), ngram_df_covered.count()))


@prefix_stack_trace()
def execute(args):
    """
    Compute differentially private set union.
    """
    spark = start_spark(sys.argv[0])
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # Uncomment the following when running locally to boost up local speed
    #spark.conf.set('spark.sql.shuffle.partitions', 2)

    if args.need_split == 1:
        extracted_all_dp_ngram = read_with_header(args.input_ngram_path, args.file_type, spark).select(F.col("token").alias("ngrams"))
        extracted_all_dp_ngram = extracted_all_dp_ngram.withColumn("size", F.size(F.split(F.col("ngrams"), " ")))
        
    # extract from each size
    for ngram_size in range(1, args.ngram_size + 1):
        log(logging.INFO, DataCategory.PUBLIC, "{}-gram...".format(ngram_size))
        input_ngram_path = os.path.join(args.input_path, "{}gram".format(ngram_size))
        input_ngram = read_with_header(input_ngram_path, args.file_type, spark)
        input_ngram = input_ngram.select(F.col("user"), F.col("ngrams"))

        if args.need_split == 0:
            extracted_dp_ngram_path = os.path.join(args.input_ngram_path, "{}gram".format(ngram_size))
            extracted_dp_ngram = read_with_header(extracted_dp_ngram_path, args.file_type, spark)
        else:
            extracted_dp_ngram = extracted_all_dp_ngram.filter(extracted_all_dp_ngram.size == ngram_size).select(F.col("ngrams"))

        log(logging.INFO, DataCategory.PUBLIC, "Loaded {} n-grams from all user".format(input_ngram.count()))
        compute_k_anon(input_ngram, extracted_dp_ngram, args)

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
