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

from .config import GraniteConfig
from .impl import (GraniteInferenceImpl, 
                   TransformersInferenceImpl, 
                   random_uuid_str, 
                   VLLMInferenceImpl,
                   GenerateResult)

# Direct import of internal symbols from other packages is standard 
# development practice within Llama Stack. 
from llama_stack.apis.inference import *  # noqa: F403
from llama_models.llama3.api.datatypes import (
    CompletionMessage, Role, StopReason, ToolCall
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper, ModelsProtocolPrivate
from llama_stack.apis.models.models import *


######################
# Constants go here
 
_TOOL_CALL_SPECIAL_TOKEN_STR = "<|tool_call|>"


_INVALID_ASSISTANT_TOKENS = (
    "<|tool_call|>", "<|start_of_role|>",
    "<|end_of_role|>", "<|end_of_text|>",
)

#####################
# Functions go here

def log(msg: str):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")
    
   
def _tools_to_json(tools: List[ToolDefinition]) -> List[Dict]:
    """
    Convert a list of tool definitions in Llama Stack's proprietary 
    JSON format into OpenAI's proprietary JSON format.
    
    :param tools: List of :class:`ToolDefinition` dataclass objects
     that describe the available tools
     
    :returns: The Python list/dictionary version of an equivalent 
     list of JSON records that describe tools in the format from 
     https://platform.openai.com/docs/guides/function-calling
    """
    result = []
    
    for tool_definition in tools:
        required_param_names = []
        properties = dict()
        for name, tool_param_definition in tool_definition.parameters.items():
            properties[name] = {
                "type": tool_param_definition.param_type,
                "description": tool_param_definition.description,
                # TODO: Translate "default" element of ToolParamDefinition, which
                #  is undocumented, with no example or schema provided.
            }
            if tool_param_definition.required:
                required_param_names.append(name)

        result.append({
            "name": tool_definition.tool_name,
            "description": tool_definition.description,
            "parameters": {
                # Undocumented JSON field that the OpenAI docs include without
                # explanation.  Our training data also uses this mystery field, 
                # so we should include it during inference.
                "type": "object",
                "properties": properties,
                "required": required_param_names
            }
        })
                
    return result
    
    
######################
# Classes go here

class _ModelResponseType(enum.Enum):
    
    EMPTY = "Insufficient output to determine output type"
    INCOMPLETE_TOOL_CALL = "Incomplete tool call"
    TOOL_CALL = "Tool call"
    INCOMPLETE_ASSISTANT_MESSAGE = "Incomplete assistant message"
    ASSISTANT_MESSAGE = "Assistant message"

@dataclasses.dataclass
class _ParsedModelResponse:
    """Internal data structure that's basically the union of
    ``ChatCompletionResponse`` and ``ChatCompletionResponseEvent``.
    
    Represents the results of parsing a prefix of the model's response
    to a prompt.
    """
    
    response_type: _ModelResponseType
    
    content: Optional[InterleavedTextMedia] = None # type: ignore
    
    tool_calls: Optional[List[ToolCall]] = None
    
class _ParseResult(enum.Enum):
    """Return codes for elements of the recursive-descent parser in 
    :class:`_ModelOutputParser`"""

    NOT_FOUND = "Did not find the target of this rule"
    INCOMPLETE = "Found an incomplete instance of the target of this rule"
    SUCCESS = "Found and parsed the target of this rule"
    
    # Errors result in an exception being thrown and no value returned
    
class _GraniteModelOutputParser:
    """Recursive-descent parser for the output of Granite models"""
    
    _tokenizer: transformers.AutoTokenizer
    
    # Integer IDs of tokens that can't appear in an assistant message
    _invalid_assistant_token_ids: Tuple[int]
    
    def __init__(self, tokenizer: transformers.AutoTokenizer) -> None:
        """
        :param tokenizer: Tokenizer associated with the model. Must match the
         model used for any subsequent parsing calls
        """
        self._tokenizer = tokenizer
        
        for t in _INVALID_ASSISTANT_TOKENS:
            if t not in self._tokenizer.vocab:
                raise ValueError(f"Tokenizer is missing expected special token '{t}'")
        
        self._invalid_assistant_token_ids = tuple(
            self._tokenizer.vocab[t] for t in _INVALID_ASSISTANT_TOKENS
        )
    
    def _parse_tool_calls(self,
                         model_response: GenerateResult,
                         position: int,
                         
                         ) -> tuple[_ParseResult, int, Optional[List[ToolCall]]]:
        """
        Attempt to parse tool calls at the indicated position.
        
        :param model_response: Output from model invocation
        :param position: Current token location from parsing left->right
        
        :returns: Tuple of:
            * Result code
            * Number of tokens consumed
            * Parse tree, which is a list of :class:`ToolCall` objects
        """
        
        if _TOOL_CALL_SPECIAL_TOKEN_STR not in self._tokenizer.vocab:
            raise ValueError(f"Cannot parse tool calls because "
                            f"'{_TOOL_CALL_SPECIAL_TOKEN_STR}' token "
                            f"is not present in model's tokenizer")
        tool_call_special_token = self._tokenizer.vocab[_TOOL_CALL_SPECIAL_TOKEN_STR]
        tool_calls = []  # Must always be a list, per API spec
        
        tokens = model_response.completion_tokens
        num_remaining_tokens = len(tokens) - position
        
        # print(f"tokens: {tokens}")
        # print(f"num_remaining_tokens: {num_remaining_tokens}")
        
        if num_remaining_tokens <= 0:
            # Empty input
            return _ParseResult.NOT_FOUND, 0, None
        if tokens[position] != tool_call_special_token:
            # Input doesn't start with tool call special token, so must not be
            # a tool call.
            return _ParseResult.NOT_FOUND, 0, None
        
        # If we get here, we found the tool call special token at the first
        # token position.
        # Tool calls must end with an "end of output" token, so first we 
        # check whether the model is done generating.
        if (model_response.stop_reason is StopReason.out_of_tokens 
            or model_response.stop_reason is None  # still streaming more tokens
            ):
            return _ParseResult.INCOMPLETE, num_remaining_tokens, None
    
        # If we get here, the model has produced a tool call special token,
        # followed by some content, followed by an end of output token.
        # Attempt to parse the content.
        tool_call_tokens = tokens[position + 1:]
        tool_call_json_str = self._tokenizer.decode(tool_call_tokens)
        try:
            tool_calls_parsed_json = json.loads(tool_call_json_str)
        except json.JSONDecodeError as e:
            # We can't assume the output of the model will be valid JSON
            raise ValueError("Error parsing tool call in model output.") from e

        if not isinstance(tool_calls_parsed_json, list):
            raise ValueError("Model output for tool calls is not a JSON list.")
        for tool_call_parsed_json in tool_calls_parsed_json:
            # Look for the "name" and "arguments" fields. 
            # Ignore any other fields that might be present.
            if "name" not in tool_call_parsed_json:
                raise ValueError(f"Model output for tool call "
                                 f"is missing required field 'name'.")
            tool_name = tool_call_parsed_json["name"]
            
            if "arguments" not in tool_call_parsed_json:
                raise ValueError(f"Model output for tool call "
                                 f"is missing required field 'arguments'.")
            tool_arguments = tool_call_parsed_json["arguments"]
            if not isinstance(tool_arguments, dict):
                raise ValueError(
                    f"Arguments for '{tool_name}' tool call in model output "
                    f"are '{json.dumps(tool_arguments)}', which is not a "
                    f"JSON object. Those arguments should have been a "
                    f"JSON object.")
            
            # According to the API spec, the individual arguments can be any
            # valid JSON data. We should probably validate this data against
            # the tools list, but that seems like a lot of work, so we'll 
            # just pass through the JSON data for now.
            
            # According to the API spec, the tool call ID must be a string.
            # No further recommendations are provided.
            # Use 32 bytes worth of random hexadecimal data because that's
            # what the Meta reference implementation does.
            tool_call_id = random_uuid_str()
            
            tool_calls.append(
                ToolCall(
                    call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=tool_arguments
                )
            )
        
        return _ParseResult.SUCCESS, num_remaining_tokens, tool_calls
    
        
    def _parse_assistant_message(
        self, model_response: GenerateResult, position: int,
    ) -> tuple[_ParseResult, int, Optional[InterleavedTextMedia]]: # type: ignore
        """
        Attempt to an assistant message at the indicated position.
        
        :param model_response: Output from model invocation
        :param position: Current token location from parsing left->right
        
        :returns: Tuple of:
            * Result code
            * Number of tokens consumed
            * Parse tree, which is the content value
        """
    
        remaining_tokens = model_response.completion_tokens[position:]
        num_remaining_tokens = len(remaining_tokens)
        
        if 0 == num_remaining_tokens:
            # Empty input
            return _ParseResult.NOT_FOUND, 0, None
        
        # Check for special tokens that shouldn't be in an assistant message
        for invalid_token_id in self._invalid_assistant_token_ids:
            if invalid_token_id in remaining_tokens:
                invalid_token_str = self._tokenizer.decode([invalid_token_id])
                raise ValueError(f"Assistant message contains invalid special token "
                                 f"'{invalid_token_str}'")
              

        # There are no other constraints on what constitutes a valid assistant 
        # message, so consume and convert the remaining model output.  
        # At the moment, the only content type the model produces is strings.
        content_str = self._tokenizer.decode(remaining_tokens)
        result_code = (
            _ParseResult.INCOMPLETE if model_response.stop_reason is None
            else _ParseResult.SUCCESS
        )
        return (result_code, num_remaining_tokens, content_str)
        
    
    def parse_model_response(self,
                             model_response: GenerateResult,
                             ) -> _ParsedModelResponse:
        """
        Top-level entry point into a recursive-descent parser for Granite
        model output.
    
        :param model_response: Results that the model has produced so far
        
        :returns: Parsed version of ``model_response``, including information
        about incomplete outputs that haven't yet been closed.
        """
        cur_token_position = 0
     
        # The response can either be a sequence of tool calls or an assistant
        # message. Check for each of these cases once.
        result, num_tokens_consumed, tool_calls = self._parse_tool_calls(
            model_response, cur_token_position
        )
        if result is _ParseResult.INCOMPLETE:
            return _ParsedModelResponse(_ModelResponseType.INCOMPLETE_TOOL_CALL)
        elif result is _ParseResult.SUCCESS:
            cur_token_position += num_tokens_consumed
            # If there's a tool call, no other content can be present
            if cur_token_position < len(model_response.completion_tokens):
                extra_content = self._tokenizer.decode(
                    model_response.completion_tokens[cur_token_position:])
                raise ValueError(f"Model emitted extra content after tool call: "
                                f"'{extra_content}'")
            return _ParsedModelResponse(
                _ModelResponseType.TOOL_CALL, tool_calls=tool_calls
            )
        
        # No tool calls decoded; look for an assistant message
        cur_token_position += num_tokens_consumed
        result, num_tokens_consumed, assistant_message_content = \
            self._parse_assistant_message(model_response, cur_token_position)
        if result is _ParseResult.INCOMPLETE:
            return _ParsedModelResponse(
                _ModelResponseType.INCOMPLETE_ASSISTANT_MESSAGE,
                content=assistant_message_content
            )
        elif result is _ParseResult.SUCCESS:
            cur_token_position += num_tokens_consumed
            if cur_token_position < len(model_response.completion_tokens):
                extra_content = self._tokenizer.decode(
                    model_response.completion_tokens[cur_token_position:])
                raise ValueError(f"Model emitted extra content after "
                                 f"assistant message: '{extra_content}'")
            return _ParsedModelResponse(
                _ModelResponseType.ASSISTANT_MESSAGE,
                content=assistant_message_content
            )
        
        # If we get here, we found nothing.
        if cur_token_position >= len(model_response.completion_tokens):
            return _ParsedModelResponse(is_empty=True)
        
        raise ValueError(f"Couldn't parse model output: '{model_response.completion_string}'")

class GraniteInferenceImpl(Inference, ModelsProtocolPrivate):
    """
    Granite model adapter for Llama Stack.
    
    Requires the configuration parameters documented in the 
    :class:`GraniteConfig` class.
    """
    
    _config: GraniteConfig
    _model_dir_path: pathlib.Path
    _impl: GraniteInferenceImpl
    
    def __init__(self, config: GraniteConfig):
        self._config = config
        self._model_dir_path = pathlib.Path(config.modeldir)
        
        print(f"Config is: {config}")
        
        if config.backend == "transformers":
            self._impl = TransformersInferenceImpl(self._model_dir_path)
        elif config.backend == "vllm":
            self._impl = VLLMInferenceImpl(self._model_dir_path)
        else:
            raise ValueError(f"Unknown backend type '{config.backend}'")

        if self._config.preload_model_name is not None:
            self._impl.preload_hook(self._config.preload_model_name)
        
    
    def _completion_to_prompt(self,
                              tokenizer: transformers.AutoTokenizer,
                              messages: List[Message], # type: ignore
                              tools: Optional[List[ToolDefinition]], # type: ignore
                              tool_choice: Optional[ToolChoice],
    ) -> str:
        """
        Internal method that converts the inputs to a chat_completion() call to
        a Granite model prompt.
        
        :param tokenizer: Tokenizer object for the current Granite model
        
        See :func:`chat_completion()` for information about other parameters.
        
        :returns: Prompt as a string
        """
        
        if tool_choice is not None and tool_choice == ToolChoice.required:
            # We need to figure out whether there is a way to prompt Granite 
            # so as to force tool usage.
            raise NotImplementedError("'required' option for tool_choice is "
                                      "not yet implemented.")
        
        # The Granite template currently runs off of thinly-disguised JSON.
        # Convert Llama objects to parsed JSON.
        messages_json = []
        for m in messages:
            # For now we pass through context values as-is without attempting
            # to reformat them for better alignment with Granite training data.
            # This approach will need to change if we add support for for other 
            # context types beyond string.
            # We may also want to detect RAG documents coming from the Llama 
            # Stack Agents layer and adjust their format.
            if isinstance(m, UserMessage) and m.context is not None:
                if not isinstance(m.context, str):
                    raise NotImplementedError(
                        f"Only string values for context are currently implemented. "
                        f"Received a value of type '{type(m.context)}'"
                    )
                content_str = m.context + m.content
            else:
                content_str = m.content
            messages_json.append({
                "role": m.role, "content": content_str
            })
        
        tools_as_parsed_json = None if tools is None else _tools_to_json(tools) 
        prompt = tokenizer.apply_chat_template(messages_json, 
                                               tools = tools_as_parsed_json, 
                                               tokenize=False, 
                                               add_generation_prompt=True) 
        return prompt
    
    
    def _generate_result_to_completion_mesage(self,
                                              parser: _GraniteModelOutputParser,
                                              generate_result: GenerateResult) -> CompletionMessage:
        """Subroutine that converts the output of the model so far into 
        Llama Stack format."""
        # Parse the returned tokens
        parsed_result = parser.parse_model_response(generate_result)
        
        # Convert the parsed model output to Llama Stack format.
        role = Role.assistant.value   # Returned role is always "Assistant"
        stop_reason = generate_result.stop_reason
        if parsed_result.response_type is _ModelResponseType.TOOL_CALL:
            # The API spec requires that the content field be a string, even
            # when there is no content. Use an empty string to avoid 100 lines 
            # of cryptic server-side error messages.
            content = ""
            tool_calls = parsed_result.tool_calls
        elif parsed_result.response_type is _ModelResponseType.ASSISTANT_MESSAGE:
            content = parsed_result.content
            # The API spec requires a list of tool calls, even for non-tool-call
            # model outputs.
            tool_calls = []
        else:
            raise ValueError(f"Unexpected model response type: "
                             f"{parsed_result.response_type}")
        
        try:
            completion_message = CompletionMessage(
                role=role,
                content=content,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )
            return completion_message
        except pydantic_core.ValidationError as e:
            # Compensate for Pydantic not telling you anything about what
            # input value triggered a given validation error.
            raise ValueError(
                f"Pydantic validation error calling CompletionMessage(\n"
                f"    role={role},\n"
                f"    content='{content}',\n"
                f"    stop_reason={stop_reason},\n"
                f"    tool_calls={tool_calls}\n"
                f")"
            ) from e
    
    async def _run_model_non_streaming(
            self, 
            model_name: str,
            tokenizer: transformers.AutoTokenizer,
            prompt: str,
            response_format: Optional[ResponseFormat],
            sampling_params: SamplingParams
        ):
        """Subroutine of :func:`chat_completion()` that handles the non-streaming
        case. This case has a completely different output format from the streaming
        case.
        """
        parser = _GraniteModelOutputParser(tokenizer)
        
        # Farm out inference to a background thread or another process, depending
        # on our choice of backend.
        model_response = await self._impl.generate(model_name, prompt, response_format,
                                                   sampling_params)
        
        # Parse the returned tokens
        parsed_result = parser.parse_model_response(model_response)
        
        # Convert the parsed model output to Llama Stack format.
        role = Role.assistant.value   # Returned role is always "Assistant"
        stop_reason = model_response.stop_reason
        if parsed_result.response_type is _ModelResponseType.TOOL_CALL:
            # The API spec requires that the content field be a string, even
            # when there is no content. Use an empty string to avoid 100 lines 
            # of cryptic server-side error messages.
            content = ""
            tool_calls = parsed_result.tool_calls
        elif parsed_result.response_type is _ModelResponseType.ASSISTANT_MESSAGE:
            content = parsed_result.content
            # The API spec requires a list of tool calls, even for non-tool-call
            # model outputs.
            tool_calls = []
        else:
            raise ValueError(f"Unexpected model response type: "
                             f"{parsed_result.response_type}")
        
        try:
            chat_completion_response = ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role=role,
                    content=content,
                    stop_reason=stop_reason,
                    tool_calls=tool_calls,
                ),
                logprobs=None
            )
            return chat_completion_response
        except pydantic_core.ValidationError as e:
            # Compensate for Pydantic not telling you anything about what
            # input value triggered a given validation error.
            raise ValueError(
                f"Pydantic validation error calling CompletionMessage(\n"
                f"    role={role},\n"
                f"    content='{content}',\n"
                f"    stop_reason={stop_reason},\n"
                f"    tool_calls={tool_calls}\n"
                f")"
            ) from e
            
    async def _run_model_streaming(
            self, 
            model_name: str,
            tokenizer: transformers.AutoTokenizer,
            prompt: str,
            response_format: Optional[ResponseFormat],
            sampling_params: SamplingParams
        ):
        """Subroutine of :func:`chat_completion()` that handles the streaming
        case. This case has a completely different output format from the 
        non-streaming case.
        """
        parser = _GraniteModelOutputParser(tokenizer)
        
        # Farm out inference to a background thread or another process, depending
        # on our choice of backend. We receive a stream of events containing 
        # generation results so far.
        result_generator = self._impl.streaming_generate(
            model_name, prompt, response_format, sampling_params
        )
        
        # Llama Stack API always starts the stream with a start event
        previous_response = _ParsedModelResponse(
            response_type=_ModelResponseType.EMPTY
        )
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta="" # Required even when empty
            )
        )
        
        async for model_response in result_generator:
            # Parse the model response so far
            parsed_response = parser.parse_model_response(model_response)
            
            # Llama Stack API expects a diff, so convert to a diff
            if parsed_response.response_type is _ModelResponseType.EMPTY:
                pass
            elif parsed_response.response_type is _ModelResponseType.INCOMPLETE_TOOL_CALL:
                # Wait until we have valid JSON
                pass
            elif parsed_response.response_type is _ModelResponseType.TOOL_CALL:
                for tool_call in parsed_response.tool_calls:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content=tool_call,
                                parse_status=ToolCallParseStatus.success
                            ),
                            stop_reason=StopReason.end_of_turn
                        ),
                    )
            elif (parsed_response.response_type is _ModelResponseType.INCOMPLETE_ASSISTANT_MESSAGE
                  or parsed_response.response_type is _ModelResponseType.ASSISTANT_MESSAGE):
                # Convert text to a diff
                prev_len = (len(previous_response.content) 
                            if previous_response.content is not None 
                            else 0)
                new_content = parsed_response.content[prev_len:]
                
                stop_reason = (StopReason.end_of_turn 
                               if parsed_response.response_type is _ModelResponseType.ASSISTANT_MESSAGE
                               else None)
                
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=new_content,
                        stop_reason=stop_reason
                    )
                )
            else:
                raise ValueError(f"Unexpected response type '{parsed_response.response_type}'")
                
            previous_response = parsed_response
        
        # Llama Stack API always ends the stream with punctuation and a final stop
        # reason
        if (parsed_response.response_type is _ModelResponseType.ASSISTANT_MESSAGE
            or parsed_response.response_type is _ModelResponseType.TOOL_CALL):
            final_stop_reason = StopReason.end_of_turn
        elif (parsed_response.response_type is _ModelResponseType.EMPTY
            or parsed_response.response_type is _ModelResponseType.INCOMPLETE_ASSISTANT_MESSAGE
            or parsed_response.response_type is _ModelResponseType.INCOMPLETE_TOOL_CALL):
            final_stop_reason = StopReason.out_of_tokens
        else:
            raise ValueError(f"Unexpected response type '{parsed_response.response_type}'")
        
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.complete,
                delta="",  # Mandatory, even when empty
                stop_reason=final_stop_reason,
            )
        )
    
    ###########################################################################
    # METHODS INHERITED FROM ModelsProtocolPrivate INTERFACE

    # The following method disappeared in the latest batch of breaking API changes.
    # Keeping it around in case the next set of breaking API changes reintroduces it.
    # async def list_models(self) -> List[ModelDef]:
    #     # Find everything that looksl ike a Granite checkpoint directory
    #     # under our base directory.
    #     checkpoint_dirs = [
    #         d for d in os.listdir(self._model_dir_path)
    #         if os.path.isdir(self._model_dir_path / d) 
    #         and os.path.exists(self._model_dir_path / d / "config.json")
    #     ]
    #     # By convention, model name is the directory name. Model name is
    #     # not currently stored in the config files as of Granite 3.0.
    #     return [
    #         ModelDef(identifier=dir_name, llama_model=dir_name) 
    #         for dir_name in checkpoint_dirs
    #     ]

    # Note that the return type of the superclass method is WRONG
    async def register_model(self, model: Model) -> Model:
        print(f"In register_model({model})")
        # Preload the model so that the first inference request won't time out
        self._impl.preload_hook(model.identifier)
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
        
        tokenizer = self._impl.tokenizer(model_id)
        
        # Convert from Llama Stack's structured representation of the conversation
        # to the Granite model's serialized string representation.
        prompt = self._completion_to_prompt(
            tokenizer, messages, tools, tool_choice
        )

        if stream:
            return self._run_model_streaming(model_id, tokenizer, prompt, 
                                             response_format, sampling_params)
        else:
            # Non-streaming
            return await self._run_model_non_streaming(model_id, tokenizer,
                                                       prompt,response_format,
                                                       sampling_params)
        
