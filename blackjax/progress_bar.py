# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Progress bar decorators for use with step functions.
Adapted from Jeremie Coullon's blog post :cite:p:`progress_bar`.
"""
from fastprogress.fastprogress import progress_bar
from jax import lax
from jax.experimental import host_callback

import re

from jax import lax
from jax.experimental import host_callback
from tqdm.auto import tqdm as tqdm_auto

_CHAIN_RE = re.compile(r"\d+$")  # e.g. get '3' from 'TFRT_CPU_3'

def progress_bar_scan(num_samples, num_chains):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    tqdm_bars = {}
    finished_chains = []
    for chain in range(num_chains):
        tqdm_bars[chain] = tqdm_auto(range(num_samples), position=chain)
        tqdm_bars[chain].set_description("Compiling.. ", refresh=True)

    def _update_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].set_description(f"Warmup {chain}", refresh=False)
        tqdm_bars[chain].update(arg)

    def _close_tqdm(arg, transform, device):
        chain_match = _CHAIN_RE.search(str(device))
        assert chain_match
        chain = int(chain_match.group())
        tqdm_bars[chain].update(arg)
        finished_chains.append(chain)
        if len(finished_chains) == num_chains:
            for chain in range(num_chains):
                tqdm_bars[chain].close()

    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(
                _update_tqdm, 0, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num_samples,
            lambda _: host_callback.id_tap(
                _close_tqdm, remainder, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )


    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return result

        return wrapper_progress_bar
    return _progress_bar_scan

