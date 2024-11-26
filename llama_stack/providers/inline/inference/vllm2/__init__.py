# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import VLLMConfig2

async def get_provider_impl(config: VLLMConfig2, _deps) -> Any:
    from .vllm2 import VLLMInferenceImpl2

    impl = VLLMInferenceImpl2(config)
    return impl
