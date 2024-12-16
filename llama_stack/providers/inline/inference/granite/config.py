# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel
from typing import Optional


class GraniteConfig(BaseModel):
    
    # Root directory where one or more models from the IBM Granite
    # family are located. Each model should be in an eponymous subdirectory.
    modeldir: str
    
    # Optional name of a Granite model to preload on startup.
    preload_model_name: Optional[str]
    
    # Name of the inference engine to use: "transformers" or "vllm"
    backend: str
    
