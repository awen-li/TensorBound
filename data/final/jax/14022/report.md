# Precision argument for matmul is not respected on Ampere GPUs like the RTX 3090

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/14022](https://github.com/jax-ml/jax/issues/14022)
- Opened: 2023-01-15T11:46:26Z
- Closed: 2023-02-08T01:13:00Z
- Labels: `bug`, `NVIDIA GPU`

## Reported behavior

### Description

As mentioned in issue https://github.com/google/jax/issues/4873, matrix multiplication defaults to using tensorFloat32 on Ampere GPUs like the RTX 3090.
However, it would be nice to be able to specify whether tensorFloat32 or float32 is used using the `precision` argument of jax numpy's matrix operations. It seeems this is currently not possible and tensorfloat32 will be used irrespective of the value of the  `precision` argument. So at the moment, it seems it is only possible to disable tensorfloat32 usage globally by setting the ` NVIDIA_TF32_OVERRIDE=0`  environment variable.

See below for a minimum reproduction example:
```
import jax
import jax.random as jrandom
import jax.numpy as jnp
import numpy as np

# compute matrix matrix product between 2 matrices of size (2, n) and (n, 2)
def compute_matrix_product(n, precision):
    key = jrandom.PRNGKey(seed=42)
    key_A, key_B = jrandom.split(key, 2)
    A = jrandom.uniform(key_A, shape=(2, n))
    B = jrandom.uniform(key_B, shape=(n, 2))
    return jnp.matmul(A, B, precision=precision)

cpu_device = jax.devices("cpu")[0]
gpu_device = jax.devices("gpu")[0]

n = 10
with jax.default_device(cpu_device):
    res_cpu = compute_matrix_product(n, precision="float32")
with jax.default_device(gpu_device):
    res_gpu_float32_precision = compute_matrix_product(n, precision="float32")
    res_gpu_tensorfloat32_precision = compute_matrix_product(n, precision="tensorfloat32")

# output difference between GPU solutions computed with different precision arguments
# expected output on Ampere GPU like the RTX 3090: finite number due to different precision between tensorfloat32 and float32
# actual output: difference on GPU with float32 and tensorfloat32 precision:  0.0
# so it looks like the same floating point format is used irrespective of the precision argument
print("difference on GPU with float32 and tensorfloat32 precision: ", np.mean(res_gpu_float32_precision - res_gpu_tensorfloat32_precision))

# check difference to CPU solutoin
# expected output:  small number on the level of float32 precision
# when run with NVIDIA_TF32_OVERRIDE=0, output is -5.9604645e-08
# actual output: difference to CPU solution:  0.00027871132
print("difference to CPU solution: ", np.mean(res_cpu - res_gpu_float32_precision))
```

### What jax/jaxlib version are you using?

jax 0.4.1, jaxlib 0.4.1+cuda11.cudnn86

### Which accelerator(s) are you using?

GPU, RTX 3090

### Additional system info

Ubuntu 20

### NVIDIA GPU info

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.32.00    Driver Version: 455.32.00    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro P620         On   | 00000000:04:00.0 Off |                  N/A |
| 34%   31C    P8    N/A /  N/A |    219MiB /  1991MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 3090    On   | 00000000:09:00.0 Off |                  N/A |
|  0%   38C    P8    26W / 350W |   2129MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Trigger

A float32 dot requests HIGHEST precision on a TF32-capable GPU while the XLA runtime executable path is enabled.

## Relevant entities

HLO dot; precision_config; LMHLO GPU GEMM; runtime custom call; GemmConfig; cuBLAS computation type

## Implemented relation

The dot operation's requested operand precision determines the computation type used by the emitted GPU GEMM.

## Pre-state

The HLO dot carries HIGHEST precision, which requires full float32 multiplication rather than TF32 input rounding.

## Binding transition

The old runtime lowering emits xla.gpu.gemm without carrying precision_config and constructs GemmConfig with the default compute precision.

## Post-state

The runtime selects TF32-as-float32 on Ampere even though the source dot requested HIGHEST precision.

## Validity obligation

Every GEMM lowering path propagates the HLO precision request to the library computation-type decision.

## Invalid decision

The runtime custom-call boundary omitted precision_config and substituted kDefaultComputePrecision.

## Physical effect

cuBLAS rounds float32 operands to TF32 mantissas for a dot that requested full float32 input precision.

## Observed consequence

The reported float32 and tensorfloat32 results are identical and differ from the CPU result by about 2.79e-4.

## Repair site

xla.gpu.gemm runtime emission and xla/service/gpu/runtime GemmConfig construction

## Repair

Commit 3febe1a serializes precision_config onto xla.gpu.gemm, decodes it in GEMM and cuBLASLt custom calls, and passes the maximum requested precision to GetBlasComputationType.

## Why the repair works

The computation-type selector now sees HIGHEST and disables TF32 for that GEMM while retaining default TF32 behavior for lower precision requests.

## Causal basis

The assignee states that the issue is fixed in the next jaxlib; the confirmed fixed build contains the assignee's same-day XLA commit whose patch propagates precision_config through the runtime path named in the workaround.

## Analysis implication

GPU lowering analysis needs the source precision_config, selected executable path, custom-call attributes, GemmConfig compute precision, and final BLAS computation type.

## Fix patches

- [`fixes/jax-14022-commit3febe1a.patch`](fixes/jax-14022-commit3febe1a.patch) — https://github.com/openxla/xla/commit/3febe1abf5ab978b84eaad014c7d8bae9e02a8cc
