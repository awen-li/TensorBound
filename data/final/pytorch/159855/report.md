# [compile] Correctness and reproducibility issue

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/159855](https://github.com/pytorch/pytorch/issues/159855)
- Opened: 2025-08-05T09:47:02Z
- Closed: 2025-09-21T13:08:13Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: ubn`

## Reported behavior

### 🐛 Describe the bug

Hey! When working on the Cache in Transformers, I stumbled upon a very worrying reproducibility (and correctness??) issue when using `torch.compile`. Full setup is slightly complicated but consider the following test:

```python
@parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
@require_read_token
def test_generation_beyond_sliding_window(self, attn_implementation: str):
    """Test that we can correctly generate beyond the sliding window. This is non trivial as
    we need to correctly slice the attention mask in all cases (because we use a HybridCache).
    Outputs for every attention functions should be coherent and identical.
    """
    model_id = "google/gemma-2-2b"
    EXPECTED_COMPLETIONS = [
        " the people, the food, the culture, the history, the music, the art, the architecture",
        ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
    ]

    input_text = [
        "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
        "A list of colors: red, blue",  # This will almost all be padding tokens
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
    inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16
    ).to(torch_device)

    # Make sure prefill is larger than sliding window
    input_size = inputs.input_ids.shape[-1]
    self.assertTrue(input_size > model.config.sliding_window)

    # Here this will trigger compilation of the forward only after prefill (first forward)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=20)[:, input_size:]

    output_text = tokenizer.batch_decode(out)
    print(output_text)
    self.assertEqual(output_text, EXPECTED_COMPLETIONS
```

Now, this model has an hybrid cache structure, i.e. it alternates between layers using full attention and layers using sliding window (local) attention.

As of now, the sliding layers update their cache in the following way when they receive new states:

```python
def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the static cache tensors in place.

    Args:
        key_states (`torch.Tensor`): The new key states to cache.
        value_states (`torch.Tensor`): The new value states to cache.
        cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

    Returns:
        tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value states.
    """
    # Lazy initialization
    if self.keys is None:
        self.lazy_initializion(key_states)

    cache_position = cache_kwargs.get("cache_position")
    self.cumulative_length += key_states.shape[-2]

    # Handle prefill phase when prompt length > sliding_window_size.
    # Note that we store cropped key/value states in the cache but return the full key/value states.
    if cache_position.shape[0] > self.max_cache_len:
        new_k = key_states[:, :, -self.max_cache_len :, :]
        new_v = value_states[:, :, -self.max_cache_len :, :]
        self.keys.copy_(new_k)
        self.values.copy_(new_v)
        return key_states, value_states

    # Sliding window logic for generation phase or prefill < window
    slicing = torch.arange(self.max_cache_len, device=self.device)
    current_seq_len = cache_position[-1] + 1  # Use last position to determine current length
    to_shift = current_seq_len > self.max_cache_len
    indices = (slicing + to_shift.sum()) % self.max_cache_len

    k_out_shifted = self.keys[:, :, indices]
    v_out_shifted = self.values[:, :, indices]

    # Clamp cache_position to determine the *target index* within the shifted cache view
    update_position = cache_position.clamp(min=0, max=self.max_cache_len - 1)

    # this weird block is for xla backends where index_copy is much better than slicing apparently
    try:
        k_out_updated = k_out_shifted.index_copy(2, update_position, key_states)
        v_out_updated = v_out_shifted.index_copy(2, update_position, value_states)
    except NotImplementedError:
        # Fallback for MPS: clone and modify the clone
        k_out_updated = k_out_shifted.clone()
        v_out_updated = v_out_shifted.clone()
        k_out_updated[:, :, update_position] = key_states
        v_out_updated[:, :, update_position] = value_states

    self.keys.copy_(k_out_updated)
    self.values.copy_(v_out_updated)
    return self.keys, self.values
```

where `self.keys`/`self.values` are pre-allocated tensors of the sliding window length.

Now is the worrying part: I simply wanted to simplify this update, instead using the following:

```python
def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the sliding window cache tensors in place.

    Args:
        key_states (`torch.Tensor`): The new key states to cache.
        value_states (`torch.Tensor`): The new value states to cache.
        cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

    Returns:
        tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value states.
    """
    # Lazy initialization
    if self.keys is None:
        self.lazy_initializion(key_states)

    cache_position = cache_kwargs.get("cache_position")

    is_full = self.cumulative_length >= self.max_cache_len
    # Update it now that we saved the value above
    self.cumulative_length += key_states.shape[-2]

    # Handle prefill phase when prompt length > sliding_window_size.
    # Note that we store cropped key/value states in the cache but return the full key/value states.
    if cache_position.shape[0] > self.max_cache_len:
        self.keys.copy_(key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(value_states[:, :, -self.max_cache_len :, :])
        # Return the full states here
        return key_states, value_states

    # Here we only assume decoding stage, i.e. 1 token at a time
    if is_full:
        self.keys.copy_(torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2))
        self.values.copy_(torch.cat((self.values[:, :, 1:, :], value_states), dim=-2))
    else:
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

    return self.keys, self.values
```

Note that in terms of logic, both implementations are completely equivalent (here we assume prefill can have an arbitrary number of tokens, but then decoding is ONLY 1 token at a time, i.e. no prefill caching or such).

With the first implementation, the test passes without any issues for all 4 attention implementations. But when switching to the second implementation, the test starts failing randomly for all except FA2 attention. But the worst part is that the results are no longer coherent between runs, i.e. 2 subsequent runs don't give the same output (note that we use `do_sample=False` in `generate`, which is greedy decoding so no randomness should play here). What's even worst, if trying to set all possible seeds explicitly, i.e.

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # needed for true deterministic
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
```

we still DON'T GET THE SAME OUTPUTS EVERY RUN.

Examples outputs from 3 subsequent runs:
```
>>> [' the people, the food, the culture, the history, the music, the art, the architecture', ', green, yellow, orange, purple, pink, black, white, brown, gray, silver']
>>> [' the people, the food, the culture, the history, the music, the art, the architecture', ', green, yellow, orange, purple, pink, white, black, brown, gray, silver']
>>> [" the people, the food, the culture, the history, the music, the art, the architecture", ", green, yellow, orange, purple, pink, brown, black, white, gray, silver"]
```

Now, if removing all `torch.compile` calls with

```python
with torch.compiler.set_stance("force_eager"):
     out = model.generate(**inputs, do_sample=False, max_new_tokens=20)[:, input_size:]
```

Everything works as expected, and the test passes consistently for all attention implementations.

So there is something going on with `torch.compile` for sure (I suspect with `cat`, but unsure). I tried playing around with the ops to try to make sense of which ops would create the issue, but without real success. Adding random `clone()` ops a bit everywhere when copying back into `self` or using `self` did not seem to help.
I do expect some slight differences for different ops due to how compile will optimize the kernels (which I first thought was why I was seeing some differences), but this should definitely be reproducible at every run. It's also quite weird that the FA2 attention implementation is not impacted, whereas all 3 other are (is compile trying to fuse some ops it should not for the other implems??)

Final note: `torch.compile` is called on the forward with args `fullgraph=True` and `mode="reduce-overhead"`, the rest is default. We also call `torch._dynamo.mark_static_address(self.keys)`, `torch._dynamo.mark_static_address(self.values)` when creating the tensors, but removing them did not help so it does not seem to be the root of the issue.
Also, note that in this test, prefill (which is not compiled) is larger than the sliding window, so all subsequent (compiled) calls to `update` are in the regime `is_full=True` <-> cache_position is a 1d tensor of a single element, with `cache_position[-1] + 1 > self.max_cache_len`



EDIT: Using `torch.roll` instead of `cat` in the following way also shows the same issue, so it does not come from the `cat` ops:

```python
def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the sliding window cache tensors in place.

    Args:
        key_states (`torch.Tensor`): The new key states to cache.
        value_states (`torch.Tensor`): The new value states to cache.
        cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

    Returns:
        tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value states.
    """
    # Lazy initialization
    if self.keys is None:
        self.lazy_initializion(key_states)

    cache_position = cache_kwargs.get("cache_position")

    is_full = self.cumulative_length >= self.max_cache_len
    # Update it now that we saved the value above
    self.cumulative_length += key_states.shape[-2]

    # Handle prefill phase when prompt length > sliding_window_size.
    # Note that we store cropped key/value states in the cache but return the full key/value states.
    if cache_position.shape[0] > self.max_cache_len:
        self.keys.copy_(key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(value_states[:, :, -self.max_cache_len :, :])
        # Return the full states here
        return key_states, value_states

    # Here we only assume decoding stage, i.e. 1 token at a time
    if is_full:
        new_keys = self.keys.roll(-1, dims=-2)
        new_keys[:, :, -1:, :] = key_states
        new_values = self.values.roll(-1, dims=-2)
        new_values[:, :, -1:, :] = value_states
        self.keys.copy_(new_keys)
        self.values.copy_(new_values)
    else:
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

    return self.keys, self.values
```


### Versions

Tested on both torch==2.6.0 with A100 hardware, and torch==2.7.1 with H100 hardware.

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @muchulee8 @amjames @aakhundov @coconutruben

## Trigger

Compiled sliding-window cache update rotates a cache and copies the rotated values back in place after multiple fusion rounds.

## Relevant entities

cache buffer; temporary rotated view; WeakDep; WAR dependency; fusion candidate; read/write index functions

## Implemented relation

A WeakDep orders a read of the pre-mutation cache contents before a write that may alias the same storage.

## Pre-state

The scheduler records that temp[:-1] reads kv_cache[1:] before kv_cache[:-1] is overwritten.

## Binding transition

Repeated fusion invokes redundant-dependency pruning and drops the WeakDep merely because the producer is already represented in the fused node.

## Post-state

The fused kernel may execute differently indexed reads and writes to the same mutated buffer without preserved write-after-read ordering.

## Validity obligation

A WAR dependency is removed during fusion only when the fused read and write are the same pointwise access over the same extent.

## Invalid decision

_prune_redundant_deps treated a WeakDep as redundant without checking whether fusable_weak_dep proved compatible indices.

## Physical effect

Threads overwrite cache elements while other accesses still need their old values, creating a data race in the fused rotation.

## Observed consequence

Deterministic model generation produces varying and incorrect completions under torch.compile.

## Repair site

torch/_inductor/scheduler.py dependency pruning and fusion legality

## Repair

PR 162316 prunes dependencies before each fusion round and retains WeakDeps unless the producer/consumer access is demonstrably pointwise-compatible; it adds an inplace flip regression.

## Why the repair works

Differently indexed aliasing reads and writes remain ordered and cannot enter the racing fused kernel.

## Causal basis

The maintainer reproduced the fused cache rotation and the patch explicitly names report 159855, the WeakDep pruning path, and the index-compatibility condition.

## Analysis implication

A fusion check needs buffer identity, mutation aliases, WAR edges, read/write index functions, extents, and the fusion round that prunes edges.

## Fix patches

- [`fixes/pytorch-159855-pr162316.patch`](fixes/pytorch-159855-pr162316.patch) — https://github.com/pytorch/pytorch/pull/162316
