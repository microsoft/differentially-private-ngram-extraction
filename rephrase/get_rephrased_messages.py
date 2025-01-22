"""
Normalize text
"""

import sys
import os.path
import argparse

import logging

# Check if we're in the local development or AML environment
if os.path.exists(os.path.join(os.path.dirname(__file__),'rephrase','utils')):
    print("In local development mode, adding common modules to path")
    # In development, add the project root to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    # Add the zip file to sys.path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root_dir, 'rephrase.zip'))


import dask.dataframe as dd
import pandas as pd
import yaml
from rephrase.utils.api.LLMApiWithResponseCache import LLMApiAAD

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
        parser = argparse.ArgumentParser(
            description="Add columns from source"
        )
    
    # add parser arguments according to inputs and outputs detailed in component_spec.yaml
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--id", type=str, required=True)    
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--output_path",  # outputs MUST begin with --output_
        required=True,
        type=str,
        help="path/filename to the output file to write the output",
    )
    

    return parser

def load_data(input_path, input_type, log):
    log.info(f"Reading from {input_path}")
    read_path = os.path.join(input_path, f"*.{input_type}")
                             
    if input_type == "csv":
        return dd.read_csv(read_path, sample_rows=10000)
    if input_type == "tsv":
        return dd.read_tsv(read_path, sample_rows=10000)
    if input_type == "parquet":
        read_path = os.path.join(input_path, f"**/*.{input_type}")
        return dd.read_parquet(read_path, engine='pyarrow', index=False)

def getMessage(text, scenarioInstructions, scenarioExamples, recent_rephrases):
    chat_request_data = [
            {"role": "system", "content": 
f"""
{scenarioInstructions}

Example Rephrased Descriptions:
{recent_rephrases}

Finally,
- Reuse wording from Example Rephrased Descriptions, when possible.
- Only return the rephrased description.
"""
            },
]
    chat_request_data.extend( scenarioExamples )
    chat_request_data.extend([
            {
                "role": "user",
                "content": text,
            },
    ])

    return chat_request_data


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def formatExamples(examples):
    transformed_data = []

    for example in examples: 
        transformed_data.append({
            "role": "user",
            "content": example.get('user', '')
        })
        transformed_data.append({
            "role": "assistant",
            "content": example.get('assistant', '')
        })
    return transformed_data


def getScenarioInstructionsOrDefault(scenarioInstructions):
    baseInstructions = scenarioInstructions["baseSystem"]
    baseExamples = formatExamples(scenarioInstructions["baseExamples"])
    
    return baseInstructions, baseExamples

def processScenario(df, column, idColumn, max_workers, endpoint, model_name, log):
    # Load the appropriate instructions and examples based on the rephrase type
    
    scenarioInstructions = read_yaml("./utils/prompts/rephrase_prompt/scenarios.yaml")
    getMessageFn = getMessage               
    
    baseInstructions, examples = getScenarioInstructionsOrDefault(scenarioInstructions)
    instructions = baseInstructions

    llmapi = LLMApiAAD(endpoint,"2023-07-01-preview", model_name, max_tokens=100, n=1, temperature=0.0, top_p=1.0, logprobs=0, batch_size=1, max_retries=3)

    message_fns = [lambda id=id, column=column: (id, getMessageFn(column, instructions, examples, llmapi.topResponses())) for id, column in zip(df[idColumn], df[column])]

    results = llmapi.parallel_post_requests_fn(message_fns, max_workers=max_workers)
    log.info(f"Finished processing rephrasing: {len(results)}")
    result_dict = {id: content for id, _, content in results}
  
    df.loc[:, column] = df[idColumn].map(result_dict)

    return df

def execute(dataset_path, column, idColumn, max_workers, output_path,endpoint,model_name, log):

    df = load_data(dataset_path, "parquet", log).compute()

    log.info(f"Loaded shape: {str(df.shape)}")
    log.info(f"df.cols: {','.join(df.columns.to_list())}")

    df_prompts = df[[column,idColumn]].drop_duplicates(subset=[column]).reset_index(drop=True)

    df_prompts = processScenario(df_prompts, column, idColumn, max_workers, endpoint, model_name, log)

    # Re-populate deduped prompts with the original data
    log.info(f"Joining data.")

    # Repopulate by joining original df with dataframe with ids and rephrased column
    df_repopulated = df.merge(df_prompts, how='left',on=idColumn, suffixes=('_original',""))

    # Refill missing values for rephrased columns.
    df_repopulated[column] = df_repopulated.groupby(by=[column+'_original'])[column].transform(lambda x: x.ffill().bfill())
    log.info(f"df_repopulated.shape: {str(df_repopulated.shape)}")
    log.info(f"df_repopulated.cols: {','.join(df_repopulated.columns.to_list())}")

    log.info(f"Storing results.")
    ddf = dd.from_pandas(df_repopulated, npartitions=100)
    ddf.to_parquet(output_path, engine='pyarrow')


def main(flags=None):
    """Script main function"""
    if not flags:
        flags = sys.argv

    module_name = flags[0]
    module_args = flags[1:]

    
    log = logging.getLogger(__name__)
    log.info("Starting Rephrasing Process")
    
    # construct the argument parser
    parser = get_arg_parser()
    args = parser.parse_args(module_args)
    args = vars(args)

    for arg in args.keys():
        log.info(
            "Argument passed: " + str(arg) + " = " + str(args[arg])
        )
    log.info("----------------")

    execute(
        args["dataset_path"], args["column"], args["id"], args["max_workers"],
        args["output_path"], args["endpoint"],args["model_name"], log
    )

if __name__ == "__main__":
    main()