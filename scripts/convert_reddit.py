"""
Converts the Reddit data (https://github.com/webis-de/webis-tldr-17-corpus, 
downloadable from https://zenodo.org/record/1043504/files/corpus-webis-tldr-17.zip) 
to the json format we use for the DPNE code
"""

import argparse
import json
import os


def convert_reddit(file_reader, file_writer, field):
    for line in file_reader:
        line = line.strip()
        items = json.loads(line)

        user = items['author']
        content = items[field]

        json_obj = dict()
        json_obj["author"] = user
        json_obj["content"] = content
        json.dump(json_obj, file_writer, ensure_ascii=False)
        file_writer.write("\n")


def get_arg_parser(parser=None):
    """
    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        argparse.ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the module
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="input file path to preprocess"
    )
    parser.add_argument(
        "--field",
        required=False,
        type=str,
        default='content',
        help="field to use as body text"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="output directory path to store preprocessed file"
    )

    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    input_path = args.input_path
    output_path = os.path.join(args.output_path, "reddit.json")

    with open(input_path, 'r', encoding='utf-8') as file_reader, open(output_path, 'w', encoding='utf-8') as file_writer:
        convert_reddit(file_reader, file_writer, args.field)
