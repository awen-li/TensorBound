# Long CUDA kernels, long overall run times and `CudnnLegacyConvRunner` related errors with `pjit` on ampere GPUs

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/8654](https://github.com/jax-ml/jax/issues/8654)
- Opened: 2021-11-22T22:17:33Z
- Closed: 2023-11-17T21:48:51Z
- Labels: `bug`, `XLA`, `NVIDIA GPU`

## Reported behavior

Using `pjit` on Ampere GPUs results in longer run times (This issue persists even if I build `jaxlib` from source with arch support explicitly enabled for `8.0` devices). I've created a small program that does distributed matmul ([multi_layer_pjit_matmul.zip](https://github.com/google/jax/files/7584617/multi_layer_pjit_matmul.zip)).  

Following is a profile of the program on Ampere GPUs captured using nsight systems. There are a lot of `cudaFree` calls and I'm not sure if they're necessary. The program takes `10 min` to run, while the actual computation happens at the very end in just a few seconds. 

![Screenshot from 2021-11-22 12-16-58](https://user-images.githubusercontent.com/4879686/142929226-e1a603d7-e9ed-4e19-a8b5-dd84bb8515f7.png)

For comparison,  here's a profile of the same program on Volta GPU:
![Screenshot from 2021-11-22 13-58-27](https://user-images.githubusercontent.com/4879686/142941387-0b6b86af-4381-4c2f-81c3-53c948594ad2.png)
 
Secondly, running another program that uses `flax` and which does spatial parallel convolution using `pjit` on Ampere devices results in the following error. Is this a known issue? Or does anyone have any idea about the underlying cause and any workarounds?

```
2021-11-22 21:32:28.481653: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481729: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481795: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481833: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481884: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481915: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:28.481952: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2086] Execution of replica 0 failed: INTERNAL: CudnnLegacyConvRunner cached across multiple StreamExecutors.
2021-11-22 21:32:34.303972: E external/org_tensorflow/tensorflow/compiler/xla/service/collective_ops_utils.h:216] This thread has been waiting for 5000ms for and may be stuck: participant NcclCliqueParticipantData{rendezvous_key=RendezvousKey{run_id=RunId: 72, global_devices=[0,1,2,3,4,5,6,7], num_local_participants=8, collective_op_kind=cross_module, op_id=9}, device_ordinal=1, stream=0x5d7bea0} waiting for all participants to arrive at rendezvous RendezvousKey{run_id=RunId: 72, global_devices=[0,1,2,3,4,5,6,7], num_local_participants=8, collective_op_kind=cross_module, op_id=9}
2021-11-22 21:32:38.481836: F external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2226] Replicated computation launch failed, but not all replicas terminated. Aborting process to work around deadlock. Failure message (there may have been multiple failures, see the error log for all failures): 

CudnnLegacyConvRunner cached across multiple StreamExecutors.
Aborted (core dumped)

```



Version details:
```
jax                              0.2.26
jaxlib                           0.1.74+cuda11.cudnn82
```

```
ii  libcudnn8                   8.2.6.15-1+cuda11.4               amd64        cuDNN runtime libraries
```


## Trigger

A spatially partitioned convolution executes one compiled XLA ConvolutionThunk across replica streams on multiple Ampere GPUs.

## Relevant entities

ConvolutionThunk; MaybeFusedConvRunner; runner cache; GPU stream; StreamExecutor; replica execution

## Implemented relation

Each cached cuDNN convolution runner is bound to the GPU stream and StreamExecutor for which it was created.

## Pre-state

One compiled convolution thunk holds a convolution configuration and is invoked for several replica streams.

## Binding transition

The old thunk passes its single member runner_cache_ to every ExecuteOnStream invocation.

## Post-state

Executions on distinct GPU streams address the same cache and may retrieve a runner created for another StreamExecutor.

## Validity obligation

Runner-cache reuse occurs only for executions on the same stream; a different stream receives a distinct runner cache.

## Invalid decision

ConvolutionThunk stored one unkeyed MaybeFusedConvRunner instead of associating cached runners with their execution streams.

## Physical effect

A replica attempts to execute a cuDNN runner carrying state from another StreamExecutor.

## Observed consequence

Replica execution reports 'CudnnLegacyConvRunner cached across multiple StreamExecutors'; remaining replicas wait at the NCCL rendezvous and the process aborts.

## Repair site

tensorflow/compiler/xla/service/gpu/convolution_thunk.{h,cc}

## Repair

Commit eb8025b removes the singleton runner cache, adds a mutex-protected map from Stream pointers to MaybeFusedConvRunner instances, and passes GetOrCreateRunner(params.stream) to each convolution execution.

## Why the repair works

Every stream now obtains its own runner object, while the mutex makes concurrent cache creation safe; no executor-specific runner is reused by another stream.

## Causal basis

The issue reports the same cross-StreamExecutor error quoted by commit eb8025b. A JAX maintainer identifies that commit as the fix for the report's second problem and later confirms its inclusion in jaxlib 0.1.75.

## Analysis implication

Execution-state analysis needs the compiled thunk, cache identity, stream identity, StreamExecutor ownership, replica, and synchronization around cache creation together.

## Fix patches

- [`fixes/jax-8654-tensorflow-eb8025b.patch`](fixes/jax-8654-tensorflow-eb8025b.patch) — https://github.com/tensorflow/tensorflow/commit/eb8025bcaf25d9ddf2ca6a7015a13b179b239c9b
