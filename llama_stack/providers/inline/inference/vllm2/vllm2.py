import dataclasses
import datetime
import json
import os
import pathlib
import pydantic_core
import enum
import concurrent.futures
from typing import AsyncGenerator, Tuple

import transformers.modeling_outputs

from .config import VLLMConfig2
from llama_stack.providers.remote.inference.vllm.vllm import build_model_aliases

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer
import llama_models.sku_list

# Direct import of internal symbols from other packages is standard 
# development practice within Llama Stack. 
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.model_registry import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    convert_message_to_dict,
    request_has_media,
)
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)

from llama_stack.apis.models.models import *
from llama_stack.apis.models import *  # noqa: F403

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import (
    BaseModelPath
)

import vllm.entrypoints.openai.protocol

######################
# Constants go here


#####################
# Functions go here

def log(msg: str):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")
    
   
def _convert_finish_reason(finish_reason: str | None) -> str | None:
    """Convert an OpenAI "finish_reason" result to the equivalent
    Llama Stack result code.
    """
    # This conversion is currently a wild guess.
    if finish_reason is None:
        return None
    elif finish_reason == "stop":
        return StopReason.end_of_turn
    else:
        return StopReason.out_of_tokens

    
    
######################
# Classes go here


class VLLMInferenceImpl2(Inference, ModelsProtocolPrivate):
    """
    vLLM-based inference model adapter for Llama Stack with support for multiple
    models.
    
    Requires the configuration parameters documented in the 
    :class:`VllmConfig2` class.
    """
    
    config: VLLMConfig2
    register_helper: ModelRegistryHelper
    model_ids: set[str]
    resolved_model_id: str | None
    engine: AsyncLLMEngine | None
    chat: OpenAIServingChat | None
    
    def __init__(self, config: VLLMConfig2):
        self.config = config 
        print(f"Config is: {self.config}")
        
        self.register_helper = ModelRegistryHelper(
            build_model_aliases())
        self.formatter = ChatFormat(Tokenizer.get_instance())
        
        # The following are initialized when paths are bound to this provider
        self._resolved_model_id = None
        self.model_ids = set()
        self._engine = None
        self._chat = None
    
    ###########################################################################
    # METHODS INHERITED FROM ModelsProtocolPrivate INTERFACE

    # Note that the return type of the superclass method is WRONG
    async def register_model(self, model: Model) -> Model:
        """
        Callback that is called when the server associates an inference endpoint
        with an inference provider.
        
        :param model: Object that encapsulates parameters necessary for identifying
         a specific LLM.
        
        :returns: The input ``Model`` object. It may or may not be permissible
         to change fields before returning this object.
        """
        print(f"In register_model({model})")
        
        # First attempt to interpret the model coordinates as a Llama model name
        resolved_llama_model = llama_models.sku_list.resolve_model(
            model.provider_model_id
        )
        if resolved_llama_model is not None:
            # Load from Hugging Face repo into default local cache dir
            resolved_model_id = resolved_llama_model.huggingface_repo
        else: # if resolved_llama_model is None
            # Not a Llama model name. Pass the model id through to vLLM's loader
            resolved_model_id = model.provider_model_id
        
        print(f"Resolved model id: {resolved_model_id}")
        
        if self._resolved_model_id is not None:
            if resolved_model_id != self._resolved_model_id:
                raise ValueError(f"Attempted to serve two LLMs (ids "
                                 f"'{self._resolved_model_id}') and "
                                 f"'{resolved_model_id}') from one copy of "
                                 f"provider '{self}'. Use multiple "
                                 f"copies of the provider instead.")
            else:
                # Model already loaded
                return model
        
        # If we get here, this is the first time registering a model.
        # Preload so that the first inference request won't time out.
        engine_args = AsyncEngineArgs(
            model=resolved_model_id,
            tokenizer=resolved_model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=self.config.enforce_eager,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Wrap the lower-level engine in an OpenAI-compatible chat API
        model_config = await self._engine.get_model_config()
        self._chat = OpenAIServingChat(
            engine_client=self._engine,
            model_config=model_config,
            base_model_paths=[
                # The layer below us will only see resolved model IDs
                BaseModelPath(resolved_model_id, resolved_model_id)],
            response_role="assistant",
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            chat_template=None,
        )
        self._resolved_model_id = resolved_model_id
        self.model_ids.add(model.model_id)
        
        print(f"Finished preloading model: {resolved_model_id}")
        
        return model
        
        
            
    ###########################################################################
    # METHODS INHERITED FROM Inference INTERFACE
    
    async def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,  # type: ignore
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        raise NotImplementedError()
    
    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],  # type: ignore
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
    
    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message], # type: ignore
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        """Main entry point. Bears a striking resemblance to the OpenAI API by
        the same name.
        
        :param model_id: String that encodes the specific model to run. By convention,
         this string is the name of a directory under a known location on the local
         filesystem. We assume that upstream code has validated this string against 
         the catalog and cleansed it of any adversarial content.
        :param messages: List of message records that describe the conversation so 
         far. The allowable message types change from time to time. As of this
         writing, the types are: :class:`UserMessage`, :class:`SystemMessage`,
         :class:`ToolResponseMessage`, and :class:`CompletionMessage`.
         CompletionMessage is actually several types of message rolled into one
         and is used for all AI responses.
        :param sampling_params: Information about how and whether to sample responses
         in a way other than top-1 greedy. How to return more than one response 
         in a single :class:`CompletionMessage` object is left as an exercise to 
         the reader.
        :param tools: A catalog of available tools as of this point in the conversation.
        :param tool_choice: A string that can be either "auto" or "required". No further
         documentation is provided. I'm guessing that "required" means that inference
         should raise an error if the model didn't return a tool call, and the model 
         should be prompted to avoid triggering such an error.
        :param tool_prompt_format: Poorly-documented mystery argument. We'll come
         back to this one.
        :param response_format: Object that encapsulates parameters for constrained
         decoding. Only JSON schemas are implemented, but users are free to specify
         a BNF grammar if they like raising NotImplementedError.
        :param stream: If ``True``, generate results in a streaming fashion, usually
         one token at a time. Completely changes required downstream code.
        
        :returns: Either the entire completion as a single object of type
         :class:`ChatCompletionResponse` dataclass, or a stream 
         of incremental results encoded with the :class:`ChatCompletionResponseStreamChunk`
         class. These classes use different names to refer to the same concepts and 
         require different control structures to consume them.  Users need two completely
         different sets of downstream code to consume the output of this API. 
         Perhaps future versions of Llama Stack will change this.
        """
        if tool_prompt_format is None:
            tool_prompt_format = ToolPromptFormat.json    
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        if model_id not in self.model_ids:
            raise ValueError(f"This adapter is not registered to model id '{model_id}'. "
                             f"Registered IDs are: {self.model_ids}")
            
        # request = ChatCompletionRequest(
        #     model=self._resolved_model_id,
        #     messages=messages,
        #     sampling_params=sampling_params,
        #     tools=tools or [],
        #     tool_choice=tool_choice,
        #     tool_prompt_format=tool_prompt_format,
        #     stream=stream,
        #     logprobs=logprobs,
        # )
        
        print(f"Messages before: {messages}")
        
        # Arguments to the vLLM call must be packaged as a dataclass.
        # Note that this dataclass has the same name as a similar dataclass in
        # Llama Stack.
        converted_messages = [
            await convert_message_to_dict(m, download=True)
            for m in messages
        ]
        
        print(f"Messages after: {converted_messages}")
        
        chat_completion_request = vllm.entrypoints.openai.protocol.ChatCompletionRequest(
            model=self._resolved_model_id,
            messages=converted_messages,
        #     sampling_params=sampling_params,
        #     tools=tools or [],
        #     tool_choice=tool_choice,
        #     tool_prompt_format=tool_prompt_format,
            stream=stream,
        #     logprobs=logprobs,
        )
        
        print(f"Converted request: {chat_completion_request}")
        
        vllm_result = await self._chat.create_chat_completion(chat_completion_request)
        print(f"Result from vLLM: {vllm_result}")
        if isinstance(vllm_result, vllm.entrypoints.openai.protocol.ErrorResponse):
            raise ValueError(f"Error from vLLM layer: {vllm_result}")
        
        # Return type depends on "stream" argument
        if stream:
            if not isinstance(vllm_result, AsyncGenerator):
                raise TypeError(f"Unexpected result type {type(vllm_result)} "
                                f"for streaming inference call")
            # vLLM client returns a stream of strings, which need to be parsed.
            # Stream comes in the form of an async generator
            return self._convert_streaming_results(vllm_result)
        else:
            if not isinstance(vllm_result, 
                              vllm.entrypoints.openai.protocol.ChatCompletionResponse):
                raise TypeError(f"Unexpected result type {type(vllm_result)} "
                                f"for non-streaming inference call")     
            return self._convert_non_streaming_results(vllm_result)
            

        
    ###########################################################################
    # INTERNAL METHODS
    
    def _convert_non_streaming_results(
        self, 
        vllm_result: vllm.entrypoints.openai.protocol.ChatCompletionResponse
    ) -> ChatCompletionResponse:
        """
        Subroutine to convert the non-streaming output of vLLM's OpenAI-compatible
        API into an equivalent Llama Stack object. 
        
        The result from vLLM's non-streaming API is a dataclass with
        the same name as the Llama Stack ChatCompletionResponse dataclass,
        but with more and different field names. We ignore the fields that
        aren't currently present in the Llama Stack dataclass.
        """       
        
        # There may be multiple responses, but we can only pass through the
        # first one.
        if len(vllm_result.choices) == 0:
            raise ValueError("Don't know how to convert response object without any "
                             "responses")
        vllm_message = vllm_result.choices[0].message
        
        converted_message = CompletionMessage(
            role=vllm_message.role,
            content=vllm_message.content,
            stop_reason=_convert_finish_reason(vllm_result.choices[0].finish_reason),
            tool_calls=[
                ToolCall(
                    call_id=t.id,
                    tool_name=t.function.name,
                    arguments=t.function.arguments
                )
                for t in vllm_message.tool_calls
            ]
        )
        
        # TODO: Convert logprobs
        
        print(f"Converted message: {converted_message}")
        
        return ChatCompletionResponse(
            completion_message=converted_message,
        )

    async def _convert_streaming_results(
        self,
        vllm_result: AsyncIterator
    ) -> AsyncIterator:
        """
        Subroutine that wraps the streaming outputs of vLLM's OpenAI-compatible
        API into a second async iterator that returns Llama Stack objects.   
        
        :param vllm_result: Stream of strings that need to be parsed     
        """
        async for chunk_str in vllm_result:
                
            # Due to OpenAI compatibility, each string in the stream
            # should start with "data: " and end with "\n\n".
            
            _PREFIX = "data: "
            _SUFFIX = "\n\n"
            if not chunk_str.startswith(_PREFIX) or not chunk_str.endswith(_SUFFIX):
                raise ValueError(f"Can't parse result string from vLLM: "
                                    f"'{re.escape(chunk_str)}'")
            
            # In between the "data: " and newlines is a JSON record
            data_str = chunk_str[len(_PREFIX):-len(_SUFFIX)]
            
            # The end of the stream is indicated with "[DONE]"
            if data_str == "[DONE]":
                return
            
            # Anything that is not "[DONE]" shoudl be JSON
            
            #print(f"Parsing JSON: {data_str}")
            
            parsed_chunk = json.loads(data_str)
            
            #print(f"Got:\n{json.dumps(parsed_chunk, indent=2)}")
            
            # Result may contain multiple completions, but Llama Stack APIs
            # only support returning one.
            first_choice = parsed_chunk["choices"][0]
            
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=first_choice["delta"]["content"],
                    stop_reason=_convert_finish_reason(
                        first_choice["finish_reason"]
                    )
                )
            )
    
    # Code temporarily duplicated with llama_stack.providers.remote.inference.vllm
    # to reduce scope of changes.
    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        options = get_sampling_options(request.sampling_params)
        if "max_tokens" not in options:
            options["max_tokens"] = self.config.max_tokens

        input_dict = {}
        media_present = request_has_media(request)
        if isinstance(request, ChatCompletionRequest):
            if media_present:
                # vllm does not seem to work well with image urls, so we download the images
                input_dict["messages"] = [
                    await convert_message_to_dict(m, download=True)
                    for m in request.messages
                ]
            else:
                input_dict["prompt"] = chat_completion_request_to_prompt(
                    request,
                    self.register_helper.get_llama_model(request.model),
                    self.formatter,
                )
        else:
            assert (
                not media_present
            ), "Together does not support media for Completion requests"
            input_dict["prompt"] = completion_request_to_prompt(
                request,
                self.register_helper.get_llama_model(request.model),
                self.formatter,
            )

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **options,
        }