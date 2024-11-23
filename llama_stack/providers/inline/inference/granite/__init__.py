# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import GraniteConfig

async def get_provider_impl(config: GraniteConfig, _deps) -> Any:
    from .granite import GraniteInferenceImpl

    impl = GraniteInferenceImpl(config)
    return impl
