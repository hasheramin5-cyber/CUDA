---
name: CUDA Error
about: Report a CUDA, nvcc, driver, or GPU runtime error
title: "[CUDA Error]: "
labels: cuda, bug
assignees: ""
---

## Project

Which CUDA project produced the error?

Example:

`05. Parallel Reduction`

## Error Type

- [ ] `nvcc` compilation error
- [ ] CUDA runtime error
- [ ] GPU/device detection error
- [ ] CUDA driver error
- [ ] Memory error
- [ ] Kernel execution error
- [ ] Other

## Environment

- Operating System:
- NVIDIA GPU:
- GPU Driver Version:
- CUDA Toolkit Version:
- `nvcc` Version:

## CUDA Information

Output of:

```bash
nvcc --version
```

```text
Paste output here
```

Output of:

```bash
nvidia-smi
```

```text
Paste output here
```

## Build Command

```bash
nvcc your_program.cu -o your_program
```

## Error Message

Paste the complete error message below.

```text
Paste error here
```

## Steps to Reproduce

1.
2.
3.

## Expected Behavior

Describe what you expected to happen.

## Actual Behavior

Describe what happened instead.

## Additional Information

Add relevant screenshots, logs, or other details here.
