"""Contains classes for querying large language models."""

""" I am trying to add Llama """
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

""" importing Llama """

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM



""" Using BAM """

from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

load_dotenv()
client = Client(credentials=Credentials.from_env())




def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    elif model_type == "llama":
        return LLAMA(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    # @abstractmethod
    # def log_probs(self, text, log_prob_range):
    #     """Returns the log probs of the text.
    #     Parameters:
    #         text: The text to get the log probs of. This can be a string or a list of strings.
    #         log_prob_range: The range of characters within each string to get the log_probs of. 
    #             This is a list of tuples of the form (start, end).
    #     Returns:
    #         A list of log probs.
    #     """
    #     pass




class LLAMA(LLM):
    """ Wrapper for Llama """
    def __init__(self, config, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.disable_tqdm = disable_tqdm


    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        
        model_id = self.config['modelID']
        config = self.config['llama_config'].copy()
        config['num_return_sequences'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None

        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

        # Set the device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)



        while response is None:
            # Encode the prompt
            encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
            try:
                outputs = model.generate(
                    encoded_prompt.input_ids,**config)
                response = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            except Exception as e:
                if 'is greater than the maximum' in str(e):
                    raise BatchSizeException()
                print(e)
                print('Retrying...')
                time.sleep(5)

        return response



class BatchSizeException(Exception):
    pass
