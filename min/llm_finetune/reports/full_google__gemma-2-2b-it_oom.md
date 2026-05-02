# Full FT OOM 로그 — google/gemma-2-2b-it

- 일시: 2026-05-03T00:59:35
- 디바이스: NVIDIA GeForce RTX 3050
- VRAM 총량: 8192 MiB
- VRAM peak (학습 중단 시점): 14550 MiB
- batch_size: 1, grad_accum: 8, max_seq: 384
- bf16: True, gradient_checkpointing: True, optimizer: adamw_bnb_8bit

## 결론
`google/gemma-2-2b-it` 모델을 RTX 3050 8GB(VRAM 8192 MiB)에서 full FT 시도 중 메모리 부족으로 학습이 중단됨. 메모리 절약 옵션(bf16 + gradient checkpointing + 8-bit Adam + batch 1)을 모두 적용한 상태에서도 불가능했으므로 이 모델은 본 환경에서 LoRA(QLoRA) 방식으로 fallback하는 것이 정당화된다.

## 예외 트레이스
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 14.21 GiB is allocated by PyTorch, and 467.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

Traceback (most recent call last):
  File "C:\workspace\longstone\min\llm_finetune\train_full.py", line 178, in main
    trainer.train()
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\transformers\trainer.py", line 2122, in train
    return inner_training_loop(
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\transformers\trainer.py", line 2527, in _inner_training_loop
    self.optimizer.step()
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\accelerate\optimizer.py", line 171, in step
    self.optimizer.step(closure)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\torch\optim\lr_scheduler.py", line 166, in wrapper
    return func.__get__(opt, opt.__class__)(*args, **kwargs)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\torch\optim\optimizer.py", line 533, in wrapper
    out = func(*args, **kwargs)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\bitsandbytes\optim\optimizer.py", line 288, in step
    self.init_state(group, p, gindex, pindex)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\bitsandbytes\optim\optimizer.py", line 474, in init_state
    state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
  File "C:\workspace\longstone\min\llm_finetune\.venv\lib\site-packages\bitsandbytes\optim\optimizer.py", line 328, in get_state_buffer
    return torch.zeros_like(p, dtype=dtype, device=p.device)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 14.21 GiB is allocated by PyTorch, and 467.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

```