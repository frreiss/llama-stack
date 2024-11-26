# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel, Field
from typing import Optional

@json_schema_type
class VLLMConfig2(BaseModel):
    
    # The default values here allow Llama 3.2-11B run on an A100.
    # Adjust according to your model and GPU.
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of tensor parallel replicas (number of GPUs to use).",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    max_model_len: int = Field(
        default=4096,
        description="Maximum context length to use during serving."
    )
    enforce_eager: bool = Field(
        default=True,
        description="Whether to use eager mode for inference (otherwise cuda graphs are used).",
    )
    gpu_memory_utilization: float = Field(
        default=0.4,
        description="What fraction of GPU memory to dedicate to each model loaded"
    )
    max_num_seqs: int = Field(
        default=16,
        description="Maximum parallel batch size for generation"
    )
