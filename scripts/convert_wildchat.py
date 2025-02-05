"""
Converts the Wildchat data (https://huggingface.co/datasets/allenai/WildChat-1M)
to the parquet format we use for the DPNE code.
"""

import dask.dataframe as dd
import pandas as pd
import random
import argparse
import os
import uuid


def process_conversations_dask(df):
    """
    Process conversation data to extract user messages and create unique identifiers using Dask.

    Parameters:
    df (DataFrame): Input Dask dataframe with conversation data

    Returns:
    DataFrame: Processed Pandas dataframe with 50,000 random rows in English
    """
    # Initialize an empty list for storing processed data
    processed_data = []

    def extract_user_messages(row):
        # Parse the conversation string into a list of dictionaries
        conversation = row['conversation'].tolist()

        # Extract user messages and their turn numbers
        for turn_dict in conversation:
            if isinstance(turn_dict, dict) and 'role' in turn_dict and turn_dict['role'] == 'user':
                # Create unique user identifier by combining header and hashed_ip
                user_identifier = f"{row['header']}_{row['hashed_ip']}"

                processed_data.append({
                    'conversation_hash': row['conversation_hash'],
                    'turn_id': row['turn'],
                    'message': turn_dict['content'],
                    'user_identifier': user_identifier,
                    'timestamp': row['timestamp'],
                    'lang': row['language']
                })

    # Apply the extraction function to each row of the Dask dataframe
    df.apply(extract_user_messages, axis=1, meta=('object')).compute()

    # Create a Pandas dataframe from the processed data
    result_df = pd.DataFrame(processed_data)

    # Filter rows to only include those in English and sample 50,000 rows with random state val set as 42 for reproducibility.
    result_df = result_df[result_df['lang'] == 'English'].sample(n=50000, random_state=42)

    return result_df


def convert_wildchat(output_path):
    """
    Converts the Wildchat data to the parquet format we use for the DPNE code.

    Parameters:
    output_path (str): Output directory path to store preprocessed file
    """
    
    # Load the Wildchat dataset using Dask and HuggingFace Hub
    df = dd.read_parquet("hf://datasets/allenai/WildChat-1M/data/train-*.parquet")

    # Process the conversation data
    processed_df = process_conversations_dask(df)

    # Generate UUIDs for each row to create a unique message identifier
    processed_df['message_id'] = [uuid.uuid4() for _ in range(len(processed_df))]
    # Convert the 'message_id' column to string before saving to Parquet
    processed_df['message_id'] = processed_df['message_id'].astype(str)
    # Print the first few rows of the processed dataframe to verify the data
    print(processed_df.head())
    
    # Print the shape of the processed dataframe to verify the number of rows and columns
    print(processed_df.shape)

    # Save the processed dataframe to a parquet file in the specified output path
    processed_df.to_parquet(os.path.join(output_path, "50k_sample.parquet"), index=False)


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

    convert_wildchat(args.output_path)

    
