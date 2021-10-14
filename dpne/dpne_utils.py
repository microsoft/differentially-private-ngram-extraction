import datetime
import time
import logging
import re
import os

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from shrike.compliant_logging import DataCategory


TOKENIZER_FIND_WHITESPACE_RE = r"[\r\n\v\f\t ]+"
APP_NAME_ROOT = "DPNE_{}_{}"

class RegularExpressionCompileError(Exception):
    """Raised when regular expression can not be compiled"""


def log(level: int, data_category: DataCategory, message: str):
    """Log the message at a given level (from the standard logging package levels: ERROR, INFO, DEBUG etc).
    Add a datetime prefix to the log message, and a SystemLog: prefix provided it is public data.
    The data_category can be one of PRIVATE or PUBLIC from shrike.compliant_logging.

    Args:
        level: logging level, best set by using logging.(INFO|DEBUG|WARNING) etc
        data_category: whether message contains private or public data
        message: message to log
    """
    message = "{}\t{}\t{}".format(datetime.datetime.now(), logging.getLevelName(level), message)
    logging.getLogger().log(level=level, msg=message, category=data_category)


def read_without_header(input_path, file_type, spark_context):
    """ Read in data as a spark dataframe. NO header.
        Inputs:
        string input_path - path to input dataset
        string file_type - file type of input dataset
        SparkSession spark_context - SparkSession instance
    """
    log(logging.INFO, DataCategory.PUBLIC, "Input file type is " + file_type)
    if file_type == "txt":
        dataframe = spark_context.read.text(input_path)
    elif file_type == "tsv":
        dataframe = spark_context.read.format("csv").option("delimiter", '\t').option("header", False).option("quote", "\u0000").load(input_path)
    elif file_type == "csv":
        dataframe = spark_context.read.format("csv").option("delimiter", ',').option("header", False).option("quote", "\u0000").load(input_path)
    else:
        log(logging.ERROR, DataCategory.PUBLIC,
            ("Input file type ", file_type, " is not supported for headerless read."))
        exit(1)
    return dataframe.fillna('')


def read_with_header(input_path, file_type, spark_context):
    """ Read in data as a spark dataframe.
        Inputs:
        string input_path - path to input dataset
        string file_type - file type of input dataset
        SparkSession spark_context - SparkSession instance
    """
    log(logging.INFO, DataCategory.PUBLIC, "Input file type is " + file_type)
    if file_type == "json":
        dataframe = spark_context.read.json(input_path, mode="DROPMALFORMED")
    elif file_type == "parquet":
        dataframe = spark_context.read.parquet(input_path).option("header", True)
    elif file_type == "tsv":
        dataframe = spark_context.read.format("csv").option("delimiter", '\t').option("header", True).option("quote", "").option("inferSchema", "true").load(input_path)
    elif file_type == "csv":
        dataframe = spark_context.read.format("csv").option("delimiter", ',').option("header", True).option("quote", "").option("inferSchema", "true").load(input_path)
    else:
        log(logging.ERROR, DataCategory.PUBLIC,
            ("Input file type ", file_type, " is not supported"))
        exit(1)
    return dataframe.fillna('')


def write(output_dataframe, output_path, file_type, header: bool, write_mode="overwrite"):
    """ Write out data with a header from a a spark dataframe.
    
    Args:
        output_dataframe (pyspark.sql.DataFrame): dataframe to write to output path
        output_path (str): path to write dataframe
        file_type (str): file type of output dataframe
        header (bool): whether to include header in output
        write_mode (str): write mode, either \"overwrite\" or \"append\"
    """
    log(logging.INFO, DataCategory.PUBLIC, "Trying to write to " + output_path)
    if file_type == "json":
        output_dataframe.write.format("json").mode(write_mode).option("header", header).save(output_path)
    elif file_type == "parquet":
        output_dataframe.write.format("parquet").mode(write_mode).option("header", header).save(output_path)
    elif file_type == "tsv":
        output_dataframe.write.format("csv").mode(write_mode).option("header", header).option("delimiter", "\t").option("quote", "\u0000").save(output_path)
    elif file_type == "csv":
        output_dataframe.write.format("csv").mode(write_mode).option("header", header).option("delimiter", ",").option("quote", "\u0000").save(output_path)
    else:
        log(logging.ERROR, DataCategory.PUBLIC,
            ("Output file type ", file_type, " is not supported"))
        exit(1)


def start_spark(module_name):
    """ Generates appname and starts a spark session, returns that session.

    Args:
        module_name (str): string module name from sys.argv[0]

    Returns:
        SparkSession: Started spark context
    """
    # Start up - get time and log
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log(logging.INFO, DataCategory.PUBLIC, start_time)
    appname = APP_NAME_ROOT.format(module_name, start_time)

    # Parse args and set up Spark Session
    spark = SparkSession \
        .builder \
        .appName(appname) \
        .getOrCreate()

    # Add zip to sparkContext so workers can find it
    if os.path.exists('smartcompose.zip'):
        spark.sparkContext.addPyFile('smartcompose.zip')
        log(logging.INFO, DataCategory.PUBLIC, "Added smartcompose.zip to PyFiles for all workers.")

    return spark


def check_and_compile_regular_expression(regex_str: str):
    """Compiles the regular expression string
    If it fails to compile, raise a RegularExpresionCompileError

    Args:
        regex_str (str): regular expression to compile

    Raises:
        RegularExpressionCompileError: if regular expression is no able to be compiled

    Returns:
        Pattern: compiled regular expression object
    """
    try:
        # According to https://docs.python.org/3/library/re.html, Python will cache most recent compiled regular expression,
        # so we don't do an implicit caching for re.compile output
        regex_compiled = re.compile(regex_str)
    except:
        raise RegularExpressionCompileError

    return regex_compiled


def string_regex_matcher(input_str: str, regex: str, replacement_str=""):
    """Python version of StringRegexMatcher in mlgtools.
    Replaces all substring matched with regular expression (regex) with replacement string (replacement_str).

    Args:
        input_str (str): input string to match
        regex (str): regular expression to match
        replacement_str (str): replacement string for string matched with regex

    Returns:
        str: string removed replacement_str if it is set, or otherwise the original string
    """
    # log error if regex is None or empty
    if not regex:
        log(logging.INFO, DataCategory.PUBLIC,
            '_string_regex_matcher: regex is None or empty. Returning original sentence.')
        return input_str

    # Compile the regular expression
    regex_compiled = check_and_compile_regular_expression(regex)

    # Return the string with replacing matched substrings with replacement_str
    return regex_compiled.sub(replacement_str, input_str)


def replace_whitespaces_to_single_whitespace(input_string: str):
    """Removes any whitespaces from the input string and replaces them with single whitespace

    Args:
        input_string (str): input string to remove whitespaces
    Returns:
        str: string without whitespaces and single whitespace replacing them
    """
    return string_regex_matcher(input_string, TOKENIZER_FIND_WHITESPACE_RE, replacement_str=" ")


def extract_ngrams_to_validate(ngram):
    """Extracts ngram that consistes with [1:] tokens (removing first token)
       to check if it exists from the n-gram extraction from previous iteration

    Args:
        ngram (str): ngrams to check
    
    Returns:
        str: ngram with [1:] tokens
    """
    if not ngram or len(ngram.split(" ")) <= 1:
        return ""
    return " ".join(ngram.split(" ")[1:])


def extract_ngrams_to_validate_tail(ngram):
    """Extracts ngram that consistes with [:-1] tokens (removing first token)
       to check if it exists from the n-gram extraction from previous iteration

    Args:
        ngram (str): ngrams to check
    
    Returns:
        str: ngram with [1:] tokens
    """
    if not ngram or len(ngram.split(" ")) <= 1:
        return ""
    return " ".join(ngram.split(" ")[:-1])


def reservoir_sample_df(user_tokens_df, Delta_0, topk=True):
    """Takes an DataFrame with (user, token) and group by each user and samples in two different ways based on topk flag,
        if it is True, select (at most) top Delta_0 most frequent n-grams that user used,
        if it is False, select (at most) uniformaly random sample Delta_0 n-grams for each user.
        token can appear multiple rows for each user if it used multiple times, so we have to aggregate the count
        to calculat top Delta_0 frequent n-grams for each user.

    Args:
        user_tokens_rdd (DataFrame): input DataFrame with userid, token column, each token has single row.
        Delta_0 (int): number of n-grams to sample from each user (at most).
        topk (bool, optional): whether to sample top K frequently used n-gram or uniformly random sample n-grams from each user.

    Returns:
        DataFrame: DataFrame with (user, token) sampled, where each token has single row. 
                    if write_count is True, there is third column (user, token, count) where count is individual contribution amount that it is precalculated.
    """
    if topk:
        # get aggregated count of each user's n-gram
        user_tokens_grouped = user_tokens_df.groupBy("user", "ngrams").agg({"*":"count"})
        # rank based on occurence, select top Delta_0
        window = Window.partitionBy(user_tokens_grouped["user"]).orderBy(user_tokens_grouped["count(1)"].desc())
        sampled = user_tokens_grouped.select("*", F.row_number().over(window).alias("row_number")).where(F.col("row_number") <= Delta_0).select("user", "ngrams")
    else:
        # find unique n-gram for each user, give random values for each n-gram
        user_tokens_unique = user_tokens_df.dropDuplicates(["user", "ngrams"]).withColumn("rnd", F.rand())
        # pick top Delta_0 with respect to the random value
        window = Window.partitionBy(user_tokens_unique["user"]).orderBy(user_tokens_unique["rnd"])
        sampled = user_tokens_unique.withColumn("row_number", F.row_number().over(window)).where(F.col("row_number") <= Delta_0).select("user", "ngrams")

    # Add a new column that has total sampled n-grams count for the user.
    # this can be done by following:
    # https://stackoverflow.com/questions/48793701/adding-a-group-count-column-to-a-pyspark-dataframe
    count_window = Window.partitionBy("user")
    # raw count is used for calculating # of users with certain n-grams (optionally to do k-anonymization as post-processing)
    sampled = sampled.select("user", "ngrams", F.sqrt(1.0 / F.count("ngrams").over(count_window)).alias("count"), F.lit(1).alias("raw_count"))
    return sampled


def count_word(rows):
    words = [word for user, word in list(rows)]
    yield len(list(set(words)))
