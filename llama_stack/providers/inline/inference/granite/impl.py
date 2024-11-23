import datetime
import dataclasses
import queue
import os
import pathlib
import asyncio
import threading
import uuid
import concurrent.futures
import abc
from typing import AsyncGenerator, Callable

import lmformatenforcer

import transformers.modeling_outputs
import vllm.sampling_params

# Deep import is mandatory for this symbol
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from .config import GraniteConfig

# Direct import of internal symbols from other packages is standard 
# development practice within Llama Stack. 
from llama_stack.apis.inference import *  # noqa: F403
from llama_models.llama3.api.datatypes import (
    CompletionMessage, Role, StopReason, ToolCall
)

import torch
import transformers
import vllm

#######################
# FUNCTIONS GO HERE

def log(msg: str):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")
    
def random_uuid_str() -> str:
    return str(uuid.uuid4().hex)

_DISABLE_HW_ACCEL = False

def _hw_accel_info() -> dict:
    """Code to set up hardware acceleration if we're going to be using
    Pytorch in-process."""
    if not _DISABLE_HW_ACCEL and torch.backends.mps.is_available():
        device_name = "mps"
        recommended_max_tokens = 100
    elif not _DISABLE_HW_ACCEL and torch.cuda.is_available():
        device_name = "cuda"
        recommended_max_tokens = 1024
    else:
        device_name = "cpu"
        recommended_max_tokens = 100
        # CPU mode; prevent thrashing
        torch.set_num_threads(4)
    return device_name, recommended_max_tokens


def _response_format_to_guided_decoding_params(
        response_format: Optional[ResponseFormat]) \
            -> vllm.sampling_params.GuidedDecodingParams:
    """
    Like Llama Stack, vLLM's OpenAI-compatible API also uses the name 
    "ResponseFormat" to describe the object that is a wrapper around
    another object that is a wrapper around another object inside 
    someone else's constrained decoding library.
    Here we translate from Llama Stack's code that stands around watching 
    other code do the work to vLLM's code that does the same. 
    Since we're interfacing with the layer of vLLM below the 
    OpenAI-compatible API, we get to skip one level of glue.
    
    :param response_format: Llama Stack version of constrained decoding
     info. Can be ``None``, indicating no constraints.
    :returns: The equivalent dataclass object for the low-level inference 
     layer of vLLM.
    """
    if response_format is None:
        return vllm.sampling_params.GuidedDecodingParams()
    
    # Llama Stack currently implements fewer types of constrained
    # decoding than vLLM does. Translate the types that exist and 
    # detect if Llama Stack adds new ones.
    if isinstance(response_format, JsonSchemaResponseFormat):
        return vllm.sampling_params.GuidedDecodingParams(
            json=response_format.json_schema
        )
    elif isinstance(response_format, GrammarResponseFormat):
        # BNF grammar.
        # Llama Stack uses the parse tree of the grammar, while vLLM
        # uses the string representation of the grammar.
        raise TypeError(f"Constrained decoding with BNF grammars is not "
                        f"currently implemented, because the reference "
                        f"implementation does not implement it.")
    else:
        raise TypeError(f"ResponseFormat object is of unexpected "
                        f"subtype '{type(response_format)}'")


def _response_format_to_transformers_callback(
    tokenizer: transformers.AutoTokenizer,
    response_format: Optional[ResponseFormat]) \
            -> Optional[Callable[[int, torch.Tensor], List[int]]]:
    """
    The Transformers library requires callers to pass in a Python callback
    in order to implement constrained decoding. This function invokes the
    existing code in ``lmformatenforcer`` to generate such callback functions
    from a declarative spec of the output format.
    
    :param tokenizer: Tokenizer object that the callback will need for
     detokenization
    :param response_format: Llama Stack version of constrained decoding
     info. Can be ``None``, indicating no constraints.
    :returns: The equivalent callback to be passed to the :func:`generate()`
     method
    """
    # The Hugging Face APIs represent constrained decoding parameters as
    # a callback. Translate to the callback function format of the same library
    # that vLLM and Llama Stack use to implement the actual constrained decoding.
    if response_format is None:
        # None means "no callback"
        return None

    if isinstance(response_format, JsonSchemaResponseFormat):
        json_schema_parser = lmformatenforcer.JsonSchemaParser(
            response_format.json_schema
        )
        callback_fn = build_transformers_prefix_allowed_tokens_fn(
                tokenizer, json_schema_parser)
        return callback_fn
    elif isinstance(response_format, GrammarResponseFormat):
        # BNF grammar.
        raise TypeError(f"Constrained decoding with BNF grammars is not "
                        f"currently implemented, because the reference "
                        f"implementation does not implement it.")
    else:
        raise TypeError(f"ResponseFormat object is of unexpected "
                        f"subtype '{type(response_format)}'")    

    
#######################
# CLASSES GO HERE

@dataclasses.dataclass
class GenerateResult:
    """
    All the things that our internal :func:`generate()` methods return,
    rolled into a dataclass for ease of maintenance.
    """
    # Not including input characters
    completion_string: str
    
    # Not including input tokens
    completion_tokens: list[int]
    
    stop_reason: StopReason


class GraniteInferenceImpl(abc.ABC):
    """
    Shared code for different versions of Granite model inference.
    """

    _model_dir: pathlib.Path
    
    # Lock for cached tokenizer, also used by subclasses for locking
    # other stuff.
    _cache_lock: threading.Lock
    _cached_tokenizer_name: str | None
    _cached_tokenizer: transformers.AutoTokenizer | None
    
    def __init__(self, model_dir: pathlib.Path):
        """
        :param model_dir: Single location on local filesystem containing 
         all models and tokenizers in eponymous subdirectories.
        """
        if not isinstance(model_dir, pathlib.Path):
            raise TypeError(f"Expected pathlib.Path for model_dir, but "
                            f"received type '{type(model_dir)}' instead.")
        self._model_dir = model_dir
        
        self._cache_lock = threading.Lock()
        self._cached_tokenizer_name = None
        self._cached_tokenizer = None
        
    def _make_model_path(self, model_name: str) -> pathlib.Path:
        model_path = pathlib.Path(self._model_dir) / model_name
        if not os.path.exists(model_path):
            raise ValueError(f"Model directory not found at {model_path}")
        return model_path
        
    def tokenizer(self, model_name: str) -> transformers.AutoTokenizer:
        """
        Load the indicated Granite model's tokenizer if it isn't already 
        loaded.
        
        :param model_name: Name of the specific Granite model variant to load
        :returns: A pointer to the loaded tokenizer object
        """
        with self._cache_lock:
            if self._cached_tokenizer_name != model_name:
                model_path = self._make_model_path(model_name)
                print(f"Loading tokenizer from {model_path}")
                self._cached_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_path
                )
                self._cached_tokenizer_name = model_name
            return self._cached_tokenizer
        
    def preload_hook(self, model_name: str):
        """
        Optional callback that triggers model loading before any other initialization
        so that subsequent inference calls won't need to wait. Implementations 
        should be thread-safe but are allowed to block.
        
        :param model_name: Name of the model, and also the directory containing
         the model.
        """
        # By default, do nothing
        pass
    
    async def shutdown_hook(self):
        """
        Optional callback that triggers resource cleanup when the server is
        shutting down.
        """
        # By default, do nothing
        pass
        
    @abc.abstractmethod
    async def generate(self, model_name: str, prompt: str,
                       response_format: Optional[ResponseFormat],
                       sampling_params: SamplingParams) -> GenerateResult:
        """
        Invoke the model for generation. Runs asynchronously but does everything
        in one big step.
        Loads the model if necessary.
        
        Returns control to the calling thread for other async tasks to 
        run.
        
        :param model_name: Name of the model, and also the directory containing
        the model.
        :param prompt: String containing the fully templated prompt to feed 
        generation.
        :param response_format: Llama Stack's proprietary way of encoding 
         the configuration for constrained decoding.
        :param sampling_params: Llama Stack's proprietary way of encoding 
         the configuration for choosing the next token
        
        :returns: Several things rolled into a single dataclass
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    async def streaming_generate(self, model_name: str, prompt: str,
                                 response_format: Optional[ResponseFormat],
                                 sampling_params: SamplingParams) \
            -> AsyncGenerator[GenerateResult, None]: # Extra type arg to make fire happy
        """
        Invoke the model for generation. Runs asynchronously and streams
        results back.
        Loads the model if necessary.
        
        Returns control to the calling thread for other async tasks to 
        run.
        
        :param model_name: Name of the model, and also the directory containing
        the model.
        :param prompt: String containing the fully templated prompt to feed 
        generation.
        :param response_format: Llama Stack's proprietary way of encoding 
         the configuration for constrained decoding.
        :param sampling_params: Llama Stack's proprietary way of encoding 
         the configuration for choosing the next token
        
        :returns: Several things rolled into a single dataclass, where the
         dataclass gets updated with all generation results so far each
         time it is invoked.
        """
        raise NotImplementedError()
 

class VLLMInferenceImpl(GraniteInferenceImpl):
    """
    Run inference over Granite models in-process, using the vLLM APIs.
    
    Requires a version of vLLM with Granite models.
    """
    
    # Cached copy of the most recently-loaded model
    _cached_model_name: str | None
    _cached_model: vllm.AsyncLLMEngine | None
    
    def __init__(self, model_dir: pathlib.Path):
        """
        :param model_dir: Single location on local filesystem containing 
         all models and tokenizers in eponymous subdirectories.
        """
        super().__init__(model_dir)
        
        # Model and tokenizer are loaded during the second phase of initialization
        # because that's how Llama Stack does things.
        self._cached_model_name = None
        self._cached_model = None
        
    def model(self, model_name: str) -> vllm.AsyncLLMEngine:
        """
        Load the indicated Granite model if it isn't already loaded.       
        Returns the current model so that the caller can use it even
        if a background thread loads a different model.
        
        :param model_name: Name of the specific Granite model variant to load
        :returns: A pointer to the loaded model object
        """
        with self._cache_lock:
            if self._cached_model_name != model_name:
                if self._cached_model_name is not None:
                    # The vLLM engine has lots of background resources that need to be
                    # manually reclaimed. We'll need to add some kind of reference 
                    # counting to allow other threads that are still using the engine
                    # to finish before shutting it down.
                    # There's also the matter of the engine sitting on most of the 
                    # GPU's memory while active. Starting a second engine before the
                    # first one has shut down could result in GPU memory exhaustion.
                    raise NotImplementedError("Switching between vLLM engines not yet implemented")
                model_path = self._make_model_path(model_name)
                print(f"Loading model from {model_path}")
                
                engine_args = vllm.AsyncEngineArgs(
                    model=model_path,
                    tokenizer=model_path,
                    
                    # Leave room for other parts of the stack
                    gpu_memory_utilization=0.5,
                )
                self._cached_model = vllm.AsyncLLMEngine.from_engine_args(engine_args)
                self._cached_model_name = model_name
            return self._cached_model
        
    def preload_hook(self, model_name: str):
        log(f"Preloading Granite model '{model_name}'")
        self.model(model_name)
        log(f"Done preloading Granite model '{model_name}'")
        
    async def shutdown_hook(self):
        with self._cache_lock:
            if self._cached_model_name is not None:
                # Assume that it's ok to cut out the ground from underneath
                # any pending inference tasks
                self._cached_model.shutdown_background_loop()
                
    async def generate(self, model_name: str, prompt: str,
                       response_format: Optional[ResponseFormat],
                       sampling_params: SamplingParams) -> GenerateResult:
        
        # vLLM generation results are independent, so we can just return the last
        # one.
        result = None
        async for result in self.streaming_generate(model_name, prompt,
                                                    response_format,
                                                    sampling_params):
            pass
        
        if result is None:
            # This case should never happen
            raise ValueError("Inference did not produce any results")
        
        return result
        
    async def streaming_generate(
        self, model_name: str, prompt: str, response_format: Optional[ResponseFormat],
        sampling_params: SamplingParams) \
            -> AsyncGenerator[GenerateResult, None]:
        model = self.model(model_name)
        tokenizer = self.tokenizer(model_name)
        
        # Convert sampling configuration from Llama Stack's proprietary format 
        # to vLLM's propietary format. vLLM can do top-p and top-k at the same time,
        # so we need to improvise a bit on the conversion.
        vllm_top_k = -1 if sampling_params.top_k == 0 else sampling_params.top_k
        
        vllm_sampling_params = vllm.SamplingParams(
            max_tokens=(None if sampling_params.max_tokens == 0
                        else sampling_params.max_tokens),
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=sampling_params.temperature,
            top_p=(sampling_params.top_p 
                   if sampling_params.strategy is SamplingStrategy.top_p
                   else 1.0),
            top_k=(vllm_top_k
                   if sampling_params.strategy is SamplingStrategy.top_k
                   else -1),
            repetition_penalty=sampling_params.repetition_penalty,
            guided_decoding=_response_format_to_guided_decoding_params(response_format)
        )
        
        # The vLLM engine requires a unique identifier for each call to generate()
        request_id = random_uuid_str()
        
        # The vLLM generate() API is streaming-only and returns an async generator.
        # The generator returns objects of type vllm.RequestOutput
        results_generator = model.generate(
            prompt, vllm_sampling_params, request_id
        )
        
        request_output: vllm.RequestOutput = None
        async for request_output in results_generator:
            # Check for weird inference failures
            if request_output.outputs is None or len(request_output.outputs) == 0:
                # This case also should never happen
                raise ValueError("Inference produced empty result")
        
            # If we get here, then request_output contains the final output of the
            # generate() call. There should be one or more output chunks.
            completion_string = "".join([output.text for output in request_output.outputs])
            completion_tokens = [
                t for output in request_output.outputs for t in output.token_ids
            ]
            
            # The final output chunk should be labeled with the reason that the 
            # overall generate() call completed.
            stop_reason_str = request_output.outputs[-1].stop_reason
            if stop_reason_str is None:
                stop_reason = None  # Still going
            elif stop_reason_str == "stop":
                stop_reason = StopReason.end_of_turn
            elif stop_reason_str == "length":
                stop_reason = StopReason.out_of_tokens
            else:
                raise ValueError(f"Unrecognized stop reason '{stop_reason_str}'")
            
            # print(f"completion string: {completion_string}")
            # print(f"stop reason: {stop_reason_str}")
            # print(f"completion tokens: {completion_tokens}")
            
            # vLLM's protocol outputs the stop token, then sets end of message
            # on the next step for some reason.
            if completion_tokens[-1] == tokenizer.eos_token_id:
                completion_tokens = completion_tokens[:-1]
                stop_reason = StopReason.end_of_message
            
            yield GenerateResult(
                completion_string=completion_string, 
                completion_tokens=completion_tokens,
                stop_reason=stop_reason
            )
            

class _TokenStreamer(transformers.generation.streamers.BaseStreamer):
    """The obvious async API for streaming token generation that the good 
    folks at Hugging Face couldn't be bothered to implement. Based loosely 
    on the ``TextIteratorStreamer`` class in the ``transformers`` library, 
    but implemented correctly and with asyncio support.
    
    Turns the callbacks from ``AutoModelForCausalLM.generate()`` into an 
    async iterator over chunks of tokens.
    
    To use, call ``generate()`` from a background thread and set the 
    ``streamer`` argument to an instance of this class.
    """
    
    # Interface between the background thread and us.
    _q: queue.Queue
    
    # Flag to indicate that we should skip the next batch of tokens because
    # they will be the prompt.
    _skip_next_token_batch: bool
    
    # Flag to indicate that the all data has been consumed.
    # We can get rid of this once we're on Python 3.13, where the Queue 
    # class has a shutdown() method.
    _done: bool
    
    # Dynamically-adjusted polling interval
    _poll_interval_sec: float
    
    _END_OF_STREAM_MARKER: str = "End of stream"
    _POLL_INTERVAL_INITIAL_VALUE = 0.01
    _POLL_INTERVAL_LINEAR_INCREASE = 0.01
    _POLL_INTERVAL_EXPONENTIAL_BACKOFF = 2.0
    
    """
    :param skip_prompt: If ``True``, discard the message where the 
    """
    def __init__(self, skip_prompt: bool = False):
        self._q = queue.Queue()
        self._done = False
        self._skip_next_token_batch = skip_prompt
        self._poll_interval_sec = _TokenStreamer._POLL_INTERVAL_INITIAL_VALUE
        
    def put(self, value):
        """
        Entry point inherited from ``BaseStreamer``
        
        :param value: Not sure what this is supposed to contain, but in practice
         it seems to be a tensor. We pass these through without modification.
        """
        if self._skip_next_token_batch:
            # Skip prompt if requested
            self._skip_next_token_batch = False
        else:
            self._q.put(value)

    def end(self):
        """Entry point inherited from ``BaseStreamer``"""
        # TODO: When Python 3.13 is our minimum version, use Queue.shutdown().
        # For now we enqueue punctuation.
        self._q.put(_TokenStreamer._END_OF_STREAM_MARKER)
        
        
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            # Extra logic because Queue.shutdown() is only available on
            # Python >= 3.13
            raise StopAsyncIteration
        
        while True:
            try:
                # Need to poll because asyncio
                value = self._q.get(block=False)
                break
            except queue.Empty:
                # Increase polling interval because no value received. 
                # TCP windowing algorithm.
                self._poll_interval_sec += _TokenStreamer._POLL_INTERVAL_LINEAR_INCREASE
                await asyncio.sleep(self._poll_interval_sec)
                continue
        
        # Decrease polling interval in response to receiving a value
        self._poll_interval_sec /= _TokenStreamer._POLL_INTERVAL_EXPONENTIAL_BACKOFF
        
        if value is _TokenStreamer._END_OF_STREAM_MARKER:  # Note pointer comparison
            self._done = True
            raise StopAsyncIteration
        return value
    
@dataclasses.dataclass
class _GenerationInputs:
    """Dataclass to encapsulate inputs for calling generate() method"""
    model: transformers.AutoModelForCausalLM
    tokenizer: transformers.AutoTokenizer
    generation_config: transformers.GenerationConfig
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]
    model_input: Dict

class TransformersInferenceImpl(GraniteInferenceImpl):
    """
    Run inference over Granite models in-process, using the Transformers APIs.
    Could be expanded to cover other LLMs that the Transformers library supports.
    Runs inference in a background thread.
    
    Requires a version of Transformers with Granite models.
    """

    # Cached copy of the most recently-loaded model
    _cached_model_name: str | None
    _cached_model: transformers.GraniteForCausalLM | None
    
    # Single background thread, wrapped in an executor for queueing
    _executor: concurrent.futures.ThreadPoolExecutor
    
    _torch_device_name: str
    _recommended_max_tokens: int
    
    def __init__(self, model_dir: pathlib.Path):
        """
        :param model_dir: Single location on local filesystem containing 
         all models and tokenizers in eponymous subdirectories.
        """
        super().__init__(model_dir)
        
        # Model and tokenizer are loaded lazily.
        self._cached_model_name = None
        self._cached_model = None  

        # Note single background thread
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        self._torch_device_name, self._recommended_max_tokens = _hw_accel_info()
    
    def model(self, model_name: str) -> transformers.GraniteForCausalLM:
        """
        Load the indicated Granite model if it isn't already loaded.       
        Returns the current model so that the caller can use it even
        if a background thread loads a different model.
        
        :param model_name: Name of the specific Granite model variant to load
        :returns: A pointer to the loaded model object
        """
        with self._cache_lock:
            if self._cached_model_name != model_name:
                model_path = self._make_model_path(model_name)
                print(f"Loading model from {model_path}")
                self._cached_model = transformers.GraniteForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16
                ).to(torch.device(self._torch_device_name))
                self._cached_model_name = model_name
            return self._cached_model
        
    def preload_hook(self, model_name: str):
        log(f"Preloading Granite model '{model_name}'")
        self.model(model_name)
        log(f"Done preloading Granite model '{model_name}'")
        
    async def generate(self, model_name: str, prompt: str,
                       response_format: Optional[ResponseFormat],
                       sampling_params: SamplingParams) -> GenerateResult:
        # Farm out the inference task to our background thread.
        concurrent_futures_future = self._executor.submit(self._handle_generate, 
            # Arguments
            model_name, prompt, response_format, sampling_params)
        async_future = asyncio.wrap_future(concurrent_futures_future)
        return await async_future
    
    async def streaming_generate(self, model_name: str, prompt: str,
                                 response_format: Optional[ResponseFormat],
                                 sampling_params: SamplingParams) \
            -> AsyncGenerator[GenerateResult, None]:
        streamer = _TokenStreamer(skip_prompt=True)
        tokenizer = self.tokenizer(model_name)
                
        # Use a single executor because we're not sure what will happen
        # if we call generate() from two threads at once.
        self._executor.submit(self._handle_streaming_generate, 
            # Arguments
            model_name, prompt, response_format, streamer, sampling_params)
        
        # Build up the tokens from the stream of callbacks
        stop_reason = None  # None means "generation hasn't stopped yet"
        generated_tokens = []
        async for tokens_tensor in streamer:
            if len(tokens_tensor.shape) != 1:
                # Defensive code for when Hugging Face introduces breaking API 
                # changes.
                raise ValueError(f"Unexpected tensor shape {tokens_tensor.shape}. "
                                 f"Should be a 1D vector")
                
            # Returned tokens are tensors which are likely still on the GPU
            # even though they're tiny and 1-dimensional.
            new_tokens = tokens_tensor.cpu().tolist()
            generated_tokens.extend(new_tokens)
            
            # The generate() method doesn't explicitly tell us why it stopped
            # generating. We are supposed to infer that from the output.
            if generated_tokens[-1] == tokenizer.eos_token_id:
                stop_reason = StopReason.end_of_turn
                # We're also supposed to strip off the end-of-turn tokens ourselves.
                generated_tokens = generated_tokens[:-1]
            
            # An astute observer might notice that we are decoding the entire output
            # here, which results in an asymptotic running time that is quadratic 
            # in the number of tokens generated.
            # Before getting upset about this, it's a good idea to reflect on the 
            # fact that the underlying generation operation is also quadratic and is 
            # much more time-consuming and runs on an expensive GPU.
            completion_string = tokenizer.decode(generated_tokens)
            yield GenerateResult(
                completion_string=completion_string,
                completion_tokens=generated_tokens,
                stop_reason=stop_reason
            )
            
        if stop_reason is None:
            # generate() stopped generating tokens. Assume that it stopped
            # because it ran into the token limit.
            yield GenerateResult(
                completion_string=completion_string,
                completion_tokens=generated_tokens,
                stop_reason=StopReason.out_of_tokens
            )
              
    def _prepare_for_generate(self, model_name: str, prompt: str,
                              response_format: Optional[ResponseFormat],
                              sampling_params: SamplingParams) -> _GenerationInputs:
        """Subroutine that encapsulates all the prerequisites
        that are necessary to call ``AutoModelForCausalLM.generate()``."""
        
        # Pull copies of the model into the current thread. Other threads
        # may load new models while we're running.
        model = self.model(model_name)
        tokenizer = self.tokenizer(model_name)
        
        # Call model.generate(). This is much harder than it should be.
        
        # Turn the conversation prefix into tokens for input to the model.
        model_input = tokenizer(
            # The conversation up to this point
            prompt,
            # Tell the tokenizer to return a tensor instead of a Python list.
            # The model expects to receive tensors as input, so you almost always
            # need to set this.
            # You must manually select a framework. Good luck switching frameworks
            # afterwards. Here we use PyTorch.
            # Enabling tensor output also CHANGES THE SHAPE OF THE OUTPUT from
            # a 1D list to a 2D singleton batch. The model can only consume batches.
            # The tokenizer's decode() method can only consume 1D lists.
            # This tensor will be created on the framework's current default 
            # device, so be sure to set that default appropriately.
            return_tensors="pt",
        )
        
        # AutoTokenizer uses two different tokenizer classes internally. One
        # of these classes has the ability to put tensors on the device they
        # should be on, while the other does not. Since we can't predict which
        # implementation we'll have, we need to assume that everything's on
        # the wrong device and move it to the right device. This of course
        # requires transforming the values under the keys of a dictionary.
        model_input = {
            k: v.to(self._torch_device_name) if isinstance(v, torch.Tensor) else v
            for k, v in model_input.items()
        }
        
        # The generate() method sometimes needs to know what is the integer ID
        # of the padding token, and for some reason this critical piece of information
        # isn't included in the serialized model. We get it from the tokenizer.
        # And of course some tokenizers don't set this parameter, in which case
        # we use the end of string token and hope for the best.
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id =  tokenizer.eos_token_id
        if pad_token_id is None:
            # Raise an error here because the some branches of the generate 
            # method won't complain about an invalid value of this parameter,
            # while others will raise a cryptic exception from deep within
            # their beam search code.
            raise ValueError(f"Couldn't figure out padding token for tokenizer {tokenizer}")

        # The supported way to pass parameters to the generate() method is 
        # to pass them to the constructor for another class, then pass the 
        # resulting object to model.generate().
        generation_config = transformers.GenerationConfig(
            max_new_tokens=(None if sampling_params.max_tokens == 0
                            else sampling_params.max_tokens),
            
            # Transformers generate() will return multiple sequences by default
            num_return_sequences=1,
            num_beams=1,
            
            # Scores are wrong anyhow, so don't bother
            output_scores=False,
            
            # If you don't set this flag, you'll get a string back instead of 
            # a collection of multiple tensors and lists.
            return_dict_in_generate=True,
            
            # VERY important parameter, with no documentation on what's the right value.
            # The right value varies by model and by application.
            # Wrong values (often including the default) will produce very bad output.
            # LOWER values result in MORE penalty for repetition, because of course they do.
            repetition_penalty = sampling_params.repetition_penalty,
            
            top_p=(sampling_params.top_p 
                   if sampling_params.strategy is SamplingStrategy.top_p
                   else 1.0),
            top_k=(sampling_params.top_k
                   if sampling_params.strategy is SamplingStrategy.top_k
                   else None),
            
            temperature=sampling_params.temperature,
            
            # See long note above.
            pad_token_id=pad_token_id,
            
            # Make sure you specify this token explicitly, or you will have
            # a bad time.
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Parameters for constrained generation are **not** passed to generate()
        # via the GenerationConfig object, but are instead passed in as a 
        # separate argument to the generate() method, containing a callback
        # that itself must contain a pointer to the model's tokenizer.
        prefix_allowed_tokens_fn=_response_format_to_transformers_callback(
            tokenizer, response_format
        )
        
        return _GenerationInputs(
            model=model, tokenizer=tokenizer,
            generation_config=generation_config,
            model_input=model_input,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )
        
    def _handle_generate(self, model_name: str, prompt: str, 
                         response_format: Optional[ResponseFormat],
                         sampling_params: SamplingParams
                         ) -> GenerateResult:
        """
        Callback for the background thread to invoke the model. Loads the model
        if necessary. See :func:`generate()` for arguments info.
        """
        print(f"In _handle_generate({model_name}, '{prompt}')")
        
        generation_inputs = self._prepare_for_generate(model_name, prompt, 
                                                       response_format,
                                                       sampling_params)
        model = generation_inputs.model
        tokenizer = generation_inputs.tokenizer
        
        model_output = model.generate(
            **(generation_inputs.model_input),
            generation_config=generation_inputs.generation_config,
            prefix_allowed_tokens_fn=generation_inputs.prefix_allowed_tokens_fn
        )
        
        # The result of generate() is of course the prompt concatenated with the 
        # additional tokens generated. Strip off the prompt.
        full_token_sequence = model_output.sequences[0].cpu().tolist()
        generated_tokens = full_token_sequence[
            len(generation_inputs.model_input["input_ids"][0]):]
        
        # The generate() method doesn't explicitly tell us why it stopped
        # generating. We are supposed to infer that from the output.
        if generated_tokens[-1] == tokenizer.eos_token_id:
            stop_reason = StopReason.end_of_turn
            # We're also supposed to strip off the end-of-turn tokens ourselves.
            generated_tokens = generated_tokens[:-1]
        else:
            stop_reason = StopReason.out_of_tokens
        
        # Of course, the model does not have a pointer to its tokenizer, so
        # we need to post-process the model's output to get a usable string.
        completion_string = tokenizer.decode(generated_tokens)
        
        return GenerateResult(
            completion_string=completion_string, 
            completion_tokens=generated_tokens,
            stop_reason=stop_reason
        )
        
    def _handle_streaming_generate(
        self, model_name: str, prompt: str, response_format: Optional[ResponseFormat],
        streamer: _TokenStreamer, sampling_params: SamplingParams
    ):
        """
        Callback for the background thread to invoke the model in streaming mode. 
        Loads the model if necessary.
        """
        generation_inputs = self._prepare_for_generate(model_name, prompt, 
                                                       response_format,
                                                       sampling_params)
        model = generation_inputs.model
        model.generate(
            **(generation_inputs.model_input),
            generation_config=generation_inputs.generation_config,
            prefix_allowed_tokens_fn=generation_inputs.prefix_allowed_tokens_fn,
            streamer=streamer
        )

    
