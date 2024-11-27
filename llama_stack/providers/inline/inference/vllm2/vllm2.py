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
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import (
    BaseModelPath
)

# These vLLM modules contain names that overlap with Llama Stack names, 
# so we import fully-qualified names
import vllm.entrypoints.openai.protocol
import vllm.sampling_params
######################
# Constants go here

# Map from Hugging Face model architecture name to appropriate tool parser.
# See vllm.entrypoints.openai.tool_parsers.ToolParserManager.tool_parsers
# for the full list of available parsers.
# TODO: Expand this list
CONFIG_TYPE_TO_TOOL_PARSER = {
    "GraniteConfig": "granite",
    "MllamaConfig": "llama3_json",
}
DEFAULT_TOOL_PARSER = "pythonic"

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


def _convert_sampling_params(sampling_params: Optional[SamplingParams],
                             response_format: Optional[ResponseFormat]) \
    -> vllm.SamplingParams:
    """Convert sampling and constrained decoding configuration from 
    Llama Stack's proprietary format to vLLM's propietary format."""
    if sampling_params is None:
        # In the absence of a user-provided sampling config, we use
        # Llama Stack defaults, which are different from vLLM defaults.
        sampling_params = SamplingParams()
    
    # vLLM can do top-p and top-k at the same time, so we need to improvise
    vllm_top_k = -1 if sampling_params.top_k == 0 else sampling_params.top_k
    vllm_sampling_params = vllm.SamplingParams.from_optional(
        max_tokens=(None if sampling_params.max_tokens == 0
                    else sampling_params.max_tokens),
        # Assume that vLLM's default stop token will work
        #stop_token_ids=[tokenizer.eos_token_id],
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
    return vllm_sampling_params
    
    
def _convert_tools(tools: Optional[List[ToolDefinition]] = None) \
            -> List[vllm.entrypoints.openai.protocol.ChatCompletionToolsParam]:
    """
    Convert the list of available tools from Llama Stack's proprietary 
    format to vLLM's version of OpenAI's proprietary format.
    """
    if tools is None:
        return []
    
    result = []
    for t in tools:
        if isinstance(t.tool_name, BuiltinTool):
            raise NotImplementedError("Built-in tools not yet implemented")
        if t.parameters is None:
            parameters = None
        else: # if t.parameters is not None
            # Convert the "required" flags to a list of required params
            required_params = [k for k, v in t.parameters.items() if v.required]
            parameters = {
                "type": "object", # Mystery value that shows up in OpenAI docs
                "properties": {
                    k: {
                        "type": v.param_type,
                        "description": v.description
                    }
                    for k, v in t.parameters.items()
                },
                "required": required_params
            }    
        
        function_def = vllm.entrypoints.openai.protocol.FunctionDefinition(
            name=t.tool_name,
            description=t.description,
            parameters=parameters
        )
        
        # Every tool definition is double-boxed in a ChatCompletionToolsParam
        result.append(
            vllm.entrypoints.openai.protocol.ChatCompletionToolsParam(
                function=function_def
            )
        )
    return result
 

    
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
        self.resolved_model_id = None
        self.model_ids = set()
        self.engine = None
        self.chat = None
    
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
        
        if self.resolved_model_id is not None:
            if resolved_model_id != self.resolved_model_id:
                raise ValueError(f"Attempted to serve two LLMs (ids "
                                 f"'{self.resolved_model_id}') and "
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
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # vLLM currently requires the user to specify the tool parser 
        # manually. To choose a tool parser, we need to determine what
        # model architecture is being used. For now, we infer that 
        # information from what config class the model uses.
        low_level_model_config = self.engine.engine.get_model_config()
        hf_config = low_level_model_config.hf_config
        hf_config_class_name = hf_config.__class__.__name__
        if hf_config_class_name in CONFIG_TYPE_TO_TOOL_PARSER:
            tool_parser = CONFIG_TYPE_TO_TOOL_PARSER[hf_config_class_name]
        else:
            # No info -- choose a default so we can at least attempt tool
            # use.
            tool_parser = DEFAULT_TOOL_PARSER
        
        # Wrap the lower-level engine in an OpenAI-compatible chat API
        model_config = await self.engine.get_model_config()
        self.chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            base_model_paths=[
                # The layer below us will only see resolved model IDs
                BaseModelPath(resolved_model_id, resolved_model_id)],
            response_role="assistant",
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            chat_template=None,
            enable_auto_tools=True,
            tool_parser=tool_parser
        )
        self.resolved_model_id = resolved_model_id
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
        if model_id not in self.model_ids:
            raise ValueError(f"This adapter is not registered to model id '{model_id}'. "
                             f"Registered IDs are: {self.model_ids}")
        
        
        # Arguments to the vLLM call must be packaged as a dataclass.
        # Note that this dataclass has the same name as a similar dataclass in
        # Llama Stack.
        converted_messages = [
            await convert_message_to_dict(m, download=True)
            for m in messages
        ]
        converted_sampling_params = _convert_sampling_params(
            sampling_params, response_format
        )
        
        #print(f"Converted sampling params:\n{converted_sampling_params}")
        
        converted_tools = _convert_tools(tools)
        
        print(f"Converted tools: {converted_tools}")
        
        # Llama will try to use built-in tools with no tool catalog, so don't enable 
        # tool choice unless at least one tool is enabled.
        converted_tool_choice = "none"
        if tool_choice == ToolChoice.auto and tools is not None and len(tools) > 0:
            converted_tool_choice = "auto"
            
        
        # TODO: Figure out what to do with the tool_prompt_format argument
        
        # TODO: Convert logprobs argument
        
        chat_completion_request = vllm.entrypoints.openai.protocol.ChatCompletionRequest(
            model=self.resolved_model_id,
            messages=converted_messages,
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            stream=stream,
        #     tool_prompt_format=tool_prompt_format,
        #     logprobs=logprobs,
        )
        
        # vLLM's OpenAI-compatible APIs take sampling parameters as multiple 
        # keyword args instead of a vLLM SamplingParams object. Copy over
        # all the parts that we currently convert from Llama Stack format.
        for param_name in ["max_tokens", "temperature", "top_p", "top_k", 
                           "repetition_penalty"]:
            setattr(chat_completion_request, param_name,
                    getattr(converted_sampling_params, param_name))

        # Guided decoding parameters are further broken out
        if converted_sampling_params.guided_decoding is not None:
            g = converted_sampling_params.guided_decoding
            chat_completion_request.guided_json = g.json
            chat_completion_request.guided_regex = g.regex
            chat_completion_request.guided_grammar = g.grammar         
        
        print(f"Converted request: {chat_completion_request}")
        
        vllm_result = await self.chat.create_chat_completion(chat_completion_request)
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
        # Tool calls come in pieces, but Llama Stack expects them in bigger
        # chunks. We build up those chunks and output them at the end.
        # This data structure holds the current set of partial tool calls.
        index_to_tool_call: Dict[int,Dict] = dict()
        
        async for chunk_str in vllm_result:    
            # Due to OpenAI compatibility, each string in the stream
            # should start with "data: " and end with "\n\n".
            _PREFIX = "data: "
            _SUFFIX = "\n\n"
            if not chunk_str.startswith(_PREFIX) or not chunk_str.endswith(_SUFFIX):
                raise ValueError(f"Can't parse result string from vLLM: "
                                    f"'{re.escape(chunk_str)}'")
            
            # In between the "data: " and newlines is an event record
            data_str = chunk_str[len(_PREFIX):-len(_SUFFIX)]
            
            # The end of the stream is indicated with "[DONE]"
            if data_str == "[DONE]":
                return
            
            # Anything that is not "[DONE]" should be JSON
            #print(f"Parsing JSON: {data_str}")
            parsed_chunk = json.loads(data_str)
            
            #print(f"Parsed JSON event to:\n{json.dumps(parsed_chunk, indent=2)}")
            
            # Result may contain multiple completions, but Llama Stack APIs
            # only support returning one.
            first_choice = parsed_chunk["choices"][0]
            converted_stop_reason = _convert_finish_reason(first_choice["finish_reason"])
            delta_record = first_choice["delta"]
                  
            #print(f"Stop reason {first_choice["finish_reason"]} => {converted_stop_reason}")
            
            if "content" in delta_record:
                # Text delta
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=delta_record["content"],
                        stop_reason=converted_stop_reason
                    )
                )
            elif "tool_calls" in delta_record:
                # Tool call(s). Buffer until we get a "tool calls" stop reason
                for tc in delta_record["tool_calls"]:
                    index = tc["index"]
                    if index not in index_to_tool_call:
                        # First time this tool call is showing up
                        print(f"First time seeing index {index} (type {type(index)})")
                        index_to_tool_call[index] = dict()
                    tool_call = index_to_tool_call[index]
                    if "id" in tc:
                        tool_call["call_id"] = tc["id"]
                    if "function" in tc:
                        if "name" in tc["function"]:
                            tool_call["tool_name"] = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            # Arguments comes in as pieces of a string
                            if "arguments_str" not in tool_call:
                                tool_call["arguments_str"] = ""
                            tool_call["arguments_str"] += tc["function"]["arguments"]
            else:
                raise ValueError(f"Don't know how to parse event delta: {delta_record}")
            
            #print(f"index_to_tool_call:\n{index_to_tool_call}")
            
            if first_choice["finish_reason"] == "tool_calls":
                # Special OpenAI code for "tool calls complete".
                # Output the buffered tool calls. Llama Stack requires a separate
                # event per tool call.
                for tool_call_record in index_to_tool_call.values():
                    # Arguments come in as a string. Parse the completed string
                    tool_call_record["arguments"] = json.loads(tool_call_record["arguments_str"])
                    del tool_call_record["arguments_str"]
                    
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content=tool_call_record,
                                parse_status="success"),
                            stop_reason=converted_stop_reason
                        )
                )
    