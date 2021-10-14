"""
Converts the mnsbc data (https://archive.ics.uci.edu/ml/datasets/msnbc.com+anonymous+web+data) 
to the json format we use for the DPNE code
"""

import argparse
import json
import os


def convert_msnbc(file_reader, file_writer):
    user_idx = 1
    starting_point = False

    for line in file_reader:
        if line and 'Sequences' in line:
            starting_point = True
            continue

        if starting_point:
            line = line.strip()
            if line:
                json_obj = dict()
                json_obj["author"] = str(user_idx)
                json_obj["content"] = line
                json.dump(json_obj, file_writer, ensure_ascii=False)
                file_writer.write("\n")
                user_idx += 1


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
    output_path = os.path.join(args.output_path, "msnbc.json")

    with open(input_path, 'r', encoding='utf-8') as file_reader, open(output_path, 'w', encoding='utf-8') as file_writer:
        convert_msnbc(file_reader, file_writer)
