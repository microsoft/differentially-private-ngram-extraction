"""
Splits input n-grams to separate folders with respect to n-gram size.
"""
import sys
import time
import os.path
import argparse
import logging

from pyspark.sql import functions as F
from shrike.compliant_logging import DataCategory, enable_compliant_logging, prefix_stack_trace

if os.path.exists('dpne.zip'):
    sys.path.insert(0, 'dpne.zip')

from dpne.dpne_utils import log, start_spark, write, read_with_header


def get_arg_parser(parser=None):
    """Parse args and do basic checks. 
    
    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        argparse.ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    
    """
    if parser is None:
        parser = argparse.ArgumentParser(description='Splits input n-grams to separate folders with respect to n-gram size')

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
        '-n',
        '--ngram_size',
        required=False,
        type=int,
        default=1,
        help='Max ngram length to extract (will extract 1 to args.ngram_size)'
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


@prefix_stack_trace()
def execute(args, module_name):
    """
    Splits the extracted n-gram files by each size. 
    """
    spark = start_spark(module_name)
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # Uncomment the following when running locally to boost up local speed
    # spark.conf.set('spark.sql.shuffle.partitions', 2)

    # Read in dataset
    dataframe = read_with_header(args.input_path, args.file_type, spark)
    # Explode the tokenized n-gram column.
    dataframe = dataframe.select(F.col("user"), F.explode(F.col("tokens")).alias("ngrams"))
    # add a column in the input dataframe with word count (ngram size)
    # https://stackoverflow.com/questions/48927271/count-number-of-words-in-a-spark-dataframe
    dataframe = dataframe.withColumn("size", F.size(F.split(F.col("ngrams"), " ")))

    for idx in range(1, args.ngram_size + 1):
        ngrams_with_size_idx = dataframe.filter(dataframe.size == idx).select(F.col("user"), F.col("ngrams"))
        ngram_out_path = os.path.join(args.output_path, "{}gram".format(idx))
        write(ngrams_with_size_idx, ngram_out_path, file_type=args.outputformat, header=True)
        log(logging.INFO, DataCategory.PUBLIC, "wrote {}-gram to {}".format(idx, ngram_out_path))    

    # Finish up
    log(logging.INFO, DataCategory.PUBLIC, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    spark.stop()


def main(flags=None):
    """ Script main function """
    if not flags:
        flags = sys.argv

    module_name = flags[0]
    module_args = flags[1:]

    # construct the argument parser
    log(logging.INFO, DataCategory.PUBLIC, "Read parameters...")
    parser = get_arg_parser()
    args = parser.parse_args(module_args)
    log(logging.INFO, DataCategory.PUBLIC, "Finished reading parameters.")

    execute(args, module_name)


if __name__ == '__main__':
    enable_compliant_logging()
    main()
