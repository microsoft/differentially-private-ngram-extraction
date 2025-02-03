from pathlib import Path
import json
import requests
from typing import Union

from ratelimit import limits

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock, local

import glob
import pandas as pd
import time
import random
from dataclasses import dataclass, field

import logging

# Cache responses
from collections import Counter, defaultdict

# Import the Azure OpenAI client function
from .azure_openai_api import get_azure_openai_client

# Event to signal all threads to pause
pause_event = Event()

# Lock to manage shared state
lock = Lock()

# Shared state to store pause duration
pause_duration = 0

class ThreadSafeResponseCache:
    def __init__(self):
        self.global_counter = Counter()  # Aggregated global counter
        self.local_counters = defaultdict(lambda: Counter())  # Per-thread counters
        self.thread_local = local()  # Initialize thread-local storage

    def topResponses(self, top_x):
        # Aggregate counts from global and all local counters
        aggregated_counter = Counter(self.global_counter)
        for local_counter in self.local_counters.values():
            aggregated_counter.update(local_counter)
        # Get the top responses
        tuples = aggregated_counter.most_common(top_x)
        prettyStr = "\n".join(["- " + element[0] for element in tuples])
        return prettyStr

    def add_response(self, content):
        # Get the thread-specific ID and update its counter
        thread_id = self._get_thread_id()
        self.local_counters[thread_id][content] += 1

    def _get_thread_id(self):
        # Ensure each thread has a unique ID
        if not hasattr(self.thread_local, 'thread_id'):
            self.thread_local.thread_id = id(self.thread_local)
        return self.thread_local.thread_id

class LLMApiAAD:
    endpoint: str
    model: str
    api_version: str
    max_tokens: int = 50
    n: int = 1
    temperature: float = 0
    top_p: float = 1
    batch_size: int = 10
    max_retries: int = 3
    prompt: str
    # Cache for storing recent responses
    response_cache: ThreadSafeResponseCache
    # response_data = {}

    STOP_SIGNAL: str = None

    TIME_PERIOD: int = 60   # time period in seconds
    CALLS: int = 50          # number of calls per time period
    
    def __init__(self, endpoint: str,api_version: str, model: str, max_tokens: int = 50, 
                 n: int = 1, temperature: float = 0, top_p: float = 1, 
                 batch_size: int = 10, max_retries: int = 3,
                 rate_calls: int = 50, rate_period: int = 60, stop_signal: Union[str, None] = None):
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model
        self.max_tokens = max_tokens
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.max_retries = max_retries

        if stop_signal is not None:
            self.STOP_SIGNAL = stop_signal
        else:
            self.STOP_SIGNAL = None

        self.TIME_PERIOD = rate_period
        self.CALLS = rate_calls

        self.response_cache = ThreadSafeResponseCache()

        self.log = logging.getLogger(__name__)

    def get_client(self):
        return get_azure_openai_client(self.api_version, self.endpoint)
    
    def call_api(self,client, messages):
        response = client.chat.completions.create(
            model=self.model,
            messages= messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n = self.n,
            top_p = self.top_p,
            stop=self.STOP_SIGNAL
        )
        return response
        
        
    def topResponses(self, top_x=100):
        responses = self.response_cache.topResponses(top_x)
        return responses
    
    def make_post_request_fn(self, client, fn, max_retries=3):
        return self.make_post_request(client, fn(), max_retries)

    def make_post_request(self,client, data, max_retries=3):
        global pause_duration

        id, messages = data

        attempts = 0
        while attempts < max_retries:
            # Wait if pause_event is set
            if pause_event.is_set():
                self.log.warning(f"Pausing for {pause_duration}")
                time.sleep(pause_duration)
                pause_event.clear()
            try:
                response = self.call_api(client=client, messages=messages)
                return id, response, response.choices[0].message.content
            except Exception as e:
                self.log.warning(f"{type(e).__name__}\: {str(e)}")
                time.sleep(15)
                attempts += 1
                continue

        self.log.warning(f"Failed after {max_retries} attempts")
        return id, None, ""

    # Submit static list of messages
    def parallel_post_requests(self, messageList, max_workers=7, max_retries=3):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submitting all the POST requests
            submissions = {executor.submit(self.make_post_request, messages, max_retries): messages for messages in messageList}
            results = self._waitResults(submissions)
        return results

    # Submit lambda functions to execute
    def parallel_post_requests_fn(self, messageListFn, max_workers=2, max_retries=3):
        results = []
        client = self.get_client()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submitting all the POST requests
            submissions = {executor.submit(self.make_post_request_fn,client, messagesFn, max_retries): messagesFn for messagesFn in messageListFn}
            results = self._waitResults(submissions)
            self.log.info(f"Received all results.")

        return results

    # Wait on threads to resolve and collect results
    def _waitResults(self, submissions):
        results = []
        totalSubmissions = len(submissions)
        completedCount = 0

        self.log.info(f"Processing {totalSubmissions} submissions to LLM")

        # Collecting the results as they complete
        for future in as_completed(submissions):
            res = submissions[future]
            completedCount += 1
            try:
                id, status_code, content = future.result(timeout=1000)
                results.append((id, status_code, content))
        
            except Exception as exc:
                self.log.info(f"Error collecting result {exc}")
                id, _ = res()
                results.append((id, None, "",None))

            if completedCount % 100 == 0:
                percentage = (completedCount / totalSubmissions) * 100
                self.log.info(f"Processed {completedCount} out of {totalSubmissions} submissions to LLM ({percentage:.2f}%)")
            
        return results
