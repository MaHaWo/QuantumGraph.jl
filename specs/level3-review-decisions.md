# Level 3 BDD Review Decisions

Human decisions recorded before Level 3 spec drafting:

- CUDA/device behavior: CPU support is always in scope. A single accelerator is optional and may be used only when available and explicitly configured.
- Unsupported accelerator settings must fail with clear backend/device errors.
- Device behavior is limited to CPU or one explicitly configured accelerator.
