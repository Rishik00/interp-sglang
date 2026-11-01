# Llama-3.2-1B decode validation

The requested configuration is a T4, FP16, 50 WildChat prompts and a 250-token
generation cap, using bounded GPU-to-CPU activation offload without compression
or activation payload writes. CUDA graphs remain enabled, with up to four
concurrent requests.

The access preflight on November 1, 2026 returned HTTP 403 for both the base and
Instruct checkpoints. Hugging Face explicitly reported that the cached account's
access request had been rejected. No alternative Modal credential was configured.
No weights were loaded and no GPU benchmark was started for this attempt.

The decode runner now forwards a local Hugging Face credential privately to both
dataset preparation and GPU workers, and also accepts an existing Modal secret.
The shared credential helper is used by both benchmark scripts. An account with
approved access is required to continue. Llama decode activation parity and
performance remain unvalidated.
