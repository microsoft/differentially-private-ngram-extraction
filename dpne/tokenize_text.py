"""
Tokenize emails with an nltk tokenizer.

Example: run the run.cmd in this folder while in current dir src/python/src
"""
import argparse
import sys
import time
import os.path
import argparse
from functools import reduce
import logging
import nltk

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf
from shrike.compliant_logging import DataCategory, enable_compliant_logging, prefix_stack_trace

if os.path.exists('dpne.zip'):
    sys.path.insert(0, 'dpne.zip')

from dpne.dpne_utils import replace_whitespaces_to_single_whitespace, log, start_spark, write, read_with_header


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
        parser = argparse.ArgumentParser(description='Tokenizes text ㅇㅁㅅㅁ and extract n-grams needed for DP n-grams extraction')

    parser.add_argument(
        '-f',
        '--file_type',
        choices=["csv", "tsv", "json", "parquet"],
        required=True,
        help='One of ["csv", "tsv", "json", "parquet"] for input file type. Only applies to local runs.')

    parser.add_argument(
        '-t',
        '--outputformat',
        required=True,
        help='Output format: json or tsv'
    )
    parser.add_argument(
        '-n',
        '--ngrams',
        required=False,
        type=int,
        default=1,
        help='Max ngram length to extract (will extract 1 to args.ngrams'
    )
    parser.add_argument(
        '-m',
        '--max_num_tokens',
        required=False,
        type=int,
        default=0,
        help='if it is set with positive number, filter emails that has larger tokens than the threshold'
    )
    parser.add_argument(
        '-a',
        "--allow_multiple_ngrams",
        required=False,
        type=int,
        default=1,
        help="if set to 1, extract all possible n-grams less or equal than n (if n set to 3, extract 1, 2, 3-grams)"
    )
    parser.add_argument(
        '-i',
        '--input_path', 
        required=True,
        type=str,
        help='path/filename to the input file')
    parser.add_argument(
        '-o',
        '--output_path',
        required=True,
        type=str,
        help='path/filename to the output file to write the output (json format)')
    
    return parser


def tokenize_words(sentence, ngram_size=1):
    """Tokenizes by whitespace with word tokens with target n_grams size.

    Args:
        sentence (string) : single sentence of email body content
        ngram_size (int, optional) : target n-gram size to extract. Defaults to 1.

    Returns:
        list(str): list of tokens 
    """

    tokens = nltk.tokenize.word_tokenize(sentence)
    ngrams = []

    if tokens:
        ngrams = list(nltk.ngrams(tokens, ngram_size))
        ngrams = [" ".join(g) for g in ngrams]
    return ngrams


def tokenize_sentences(post, max_num_tokens=0):
    """Tokenizes body text into sentences, replace any whitespaces into single whitespace.

    Args:
        post (string) : body text content. 
        max_num_tokens (int, optional): if it is set with a positive number, filter out emails with tokens more than this number. Defaults to 0.

    Returns:
        list(str): list of clenaed sentences
    """
    if post == '' or post is None:
        return []

    # if max_num_tokens is set, filter out emails with tokens more than the number, token calculation is simply done with whitespace tokenization.
    if max_num_tokens > 0 and len(post.split()) > max_num_tokens:
        return []

    sentences = nltk.tokenize.sent_tokenize(post)
    # remove multiple whitespaces in each sentence text
    sentences = [replace_whitespaces_to_single_whitespace(sentence) for sentence in sentences]
    return sentences


def tokenize_df(input_df, ngram_size=1, max_num_tokens=0, allow_multiple_ngrams=True, remove_euii_tokens=True):
    """Tokenizes each row of input_df DataFrame and returns a DataFrame with (user, token) where token column exploded.

    Args:
        input_df (DataFrame): input DataFrame with "author" column with user email address, "content" column with email body text.
        ngram_size (int, optional): target ngram size to extract. Defaults to 1.
        max_num_tokens (int, optional): if it is set with a positive number, filter out emails with tokens more than this number. Defaults to 0.
        allow_multiple_ngrams (bool, optional): if True, extract all possible n-grams less or equal than n (if n set to 3, extract 1, 2, 3-grams). Defaults to True.
        remove_euii_tokens (bool, optional): if True, do not extract any n-grams with EUII scrubbed special tokens. Defaults to True.

    Returns:
        DataFrame: DataFrame with (user, list of n-grams) rows, each rows has n-grams from a single sentence in email
                (so for each email, there will be n rows if it has n sentences)
    """
    # to concatenate all dataframe row-wise
    # https://datascience.stackexchange.com/questions/11356/merging-multiple-data-frames-row-wise-in-pyspark
    def union_all(*dfs):
        return reduce(DataFrame.unionAll, dfs)

    tokenize_word_udf = udf(tokenize_words, ArrayType(StringType()))
    tokenize_sentence_udf = udf(tokenize_sentences, ArrayType(StringType()))

    tokenized_sentences = input_df.select(F.col("author").alias("user"), F.explode(tokenize_sentence_udf("content", F.lit(max_num_tokens))).alias("sentences"))
    if allow_multiple_ngrams:
        tokenized_words_for_each_size = []
        for cur_ngram_size in range(1, ngram_size+1):
            tokenized_words_for_each_size.append(tokenized_sentences.select(F.col("user"), tokenize_word_udf("sentences", F.lit(cur_ngram_size)).alias("tokens")))
        tokenized_words = union_all(*tokenized_words_for_each_size)
    else:
        tokenized_words = tokenized_sentences.select(F.col("user"), tokenize_word_udf("sentences", F.lit(ngram_size)).alias("tokens"))
    return tokenized_words


def tokenize_emails(input_data, args):
    """
    tokenizes each email with SmartCompose tokenizer and output tokens
    """
    n_grams = args.ngrams
    max_num_tokens = args.max_num_tokens
    allow_multiple_ngrams = args.allow_multiple_ngrams == 1
    tokenized = tokenize_df(input_data, n_grams, max_num_tokens, allow_multiple_ngrams)
    results = tokenized if len(tokenized.select(F.col("user")).head(1)) > 0 else None
    return results


@prefix_stack_trace()
def execute(args, module_name):
    """
    Tokenize emails and generate n-grams to use on DP n-grams extraction.
    """
    spark = start_spark(sys.argv[0])
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # Uncomment the following when running locally to boost up local speed
    #spark.conf.set('spark.sql.shuffle.partitions', 2)

    # Read in dataset
    dataframe = read_with_header(args.input_path, args.file_type, spark)
    log(logging.INFO, DataCategory.PUBLIC, "Loaded {} text bodies".format(dataframe.count()))

    outputdata = tokenize_emails(dataframe, args)

    # Write output
    if outputdata:
        log(logging.INFO, DataCategory.PUBLIC, "Stored {} tokenized sentences".format(outputdata.count()))
        write(outputdata, args.output_path, file_type=args.outputformat, header=True)
    else:
        log(logging.INFO, DataCategory.PUBLIC_DATA, "No text bodies have at least 1 n-gram")
        # write nothing if nothing yielded
        file_writer = open(os.path.join(args.output_path, "empty.{}".format(args.outputformat)), 'w', encoding='utf-8')
        file_writer.close()

    # Finish up
    log(logging.INFO, DataCategory.PUBLIC, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    spark.stop()


def main(flags=None):
    """ Script main function """
    if not flags:
        flags = sys.argv

    module_name = flags[0]
    module_args = flags[1:]

    log(logging.INFO, DataCategory.PUBLIC, "Read parameters...")

    # construct the argument parser
    parser = get_arg_parser()
    args = parser.parse_args(module_args)

    log(logging.INFO, DataCategory.PUBLIC, "Finished reading parameters.")

    execute(args, module_name)


if __name__ == '__main__':
    enable_compliant_logging()
    main()
