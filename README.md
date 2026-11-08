# mini-golfhe

An 18-hole mini-golf game whose shots are simulated under fully homomorphic encryption.

## Requirements

- [Rust](https://rustup.rs) (rustup installs the pinned toolchain)
- Node.js 22 or later
- 16 GB of RAM
- Optional GPU: the CUDA toolkit on NVIDIA, ROCm on AMD

## Run

```sh
npm --prefix web install
npm run dev
```

Then open http://localhost:4321.

The server picks the best hardware it finds: CUDA on NVIDIA, HIP on AMD, Metal or Vulkan on other GPUs, otherwise the fastest CPU backend (`MINIGOLFHE_GPU=none` forces the CPU).

## Security

The parameters target 128 bits of security: every ciphertext and key fits the modulus bound of its secret in the 128-bit table of [sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/aa2c40d4eee4551e6fba06c55d7432f6918e1597/Precomputed-Tables/128bits_security.md), at `N = 2^16` (1707 bits for the dense secret of Hamming weight 1024, 345 bits for the sparse bootstrapping secret of weight 32).

## Accuracy

The encrypted shot follows the plaintext model, with minor differences. Both find contacts on the same 20 ms grid, but the encrypted comparisons are blurred within a few millimetres of each face, so a contact can come one step late and its state a few millimetres off. On grazing, corner and slow contacts the blur can also resolve the contact differently: about 3% of random shots then come to rest 5 cm to 1 m away from where the plaintext model puts them, and CKKS noise occasionally adds such a case. The client's `--scheme plaintext` option computes the contacts in the clear, for comparison.
