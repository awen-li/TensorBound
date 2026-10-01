# Using triton.autotune with the kernel updated in place will result in incorrect output

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/6547](https://github.com/triton-lang/triton/issues/6547)
- Opened: 2025-04-21T12:42:47Z
- Closed: 2025-04-24T07:48:02Z
- Labels: `bug`

## Reported behavior

### Describe the bug

Using triton.autotune with the kernel updated in place will result in incorrect output。
In the following code, the kernel updates logits_ptr in place to avoid allocating extra memory. When using triton.autotune, the kernel is executed multiple times, which can cause logits_ptr to become incorrect. Adding the reset_to_zero parameter will set logits_ptr to 0, resulting in an error.


``` python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
    ],
    key=['rows', 'cols'],
    reset_to_zero=['logits_ptr'],
    restore_value=['logits_ptr']
)

@triton.jit
def inplace_scale_2d_kernel(
    logits_ptr,
    logits_row_stride,
    scaling_penalties_ptr,
    penalties_row_stride,
    rows, cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets[:, None] < rows  # [BLOCK_SIZE, 1]
    col_mask = col_offsets[None, :] < cols  # [1, BLOCK_SIZE]
    mask = row_mask & col_mask              # [BLOCK_SIZE, BLOCK_SIZE]

    logits_offsets = row_offsets[:, None] * logits_row_stride + col_offsets[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=mask)

    penalties_offsets = row_offsets[:, None] * penalties_row_stride + col_offsets[None, :],
    penalties = tl.load(scaling_penalties_ptr + penalties_offsets, mask=mask)

    condition = logits > 0
    scaled = logits / tl.where(condition, penalties, 1.0/penalties)
    tl.store(logits_ptr + logits_offsets, scaled, mask=mask)
```

### Environment details

triton                            3.1.0
A800GPU

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The autotuner executes multiple configurations of an in-place scaling kernel against the same logits buffer.

## Observed implementation decision

Repeated benchmark executions observe state produced by prior candidate executions unless the caller supplies a suitable restore policy.

## Physical effect

Candidate configurations scale already modified logits instead of one common initial value.

## Observed consequence

Autotuning produces an incorrect final output for the in-place buffer.

## Reported proposal or workaround

The report mentions reset_to_zero and restore_value controls but does not establish their intended combination for this mutation.

## Repair trace

The issue was closed as not planned without maintainer analysis, contract clarification, implementation change, or fixed regression. The available record cannot distinguish a tool defect from unsupported stateful-autotune use or identify a repair.

## Archived evidence

- Raw report: `data/raw/github/triton/6547/report.json`
- SHA-256: `865000cd35bfbe5199783062c6be8bc2243fcda7a0e43d1614dcfd41e18e0077`
