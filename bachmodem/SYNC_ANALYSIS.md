# BachModem GPU Synchronization Analysis

## Overview
This document tracks GPU→CPU synchronization points that break the computation stream in the BachModem decoder pipeline. These are bottlenecks where the CPU must wait for the GPU.

## ✅ Fixed Synchronization Points

### 1. **deinterleave_gpu.rs** 
- **Status**: ✅ FIXED
- **Issue**: Test used `.to_data()` for validation
- **Solution**: Created `gpu_test_utils::validate_roundtrip()` for GPU-only testing

### 2. **wav.rs**
- **Status**: ✅ OPTIMIZED
- **Issue**: `.into_data()` at line 18 forced sync during generation
- **Solution**: 
  - Added `prepare_wav_signal_gpu()` for normalization on GPU
  - Kept `write_wav()` as final sync point (required for file I/O)
  - Documented with ⚠️ warnings

### 3. **modulation.rs - Synchronization**
- **Status**: ✅ IMPROVED
- **Issue**: Lines 135-140: `.into_scalar()` calls in `synchronize_signal()`
- **Solution**: 
  - Created `synchronize_signal_gpu()` that returns tensors
  - Original function wraps it for backwards compatibility
  - Documented sync points

### 4. **gpu_ops.rs - SNR Estimation**
- **Status**: ✅ IMPROVED
- **Issue**: Lines 126, 155: `.into_scalar()` in `estimate_snr_from_correlation()`
- **Solution**:
  - Created `estimate_snr_from_correlation_gpu()` returning tensor SNR
  - Computation stays on GPU (using `ln(x)/ln(10)` instead of `.log10()`)
  - Original function wraps for compatibility

### 5. **rake.rs - Peak Detection**
- **Status**: ⚠️ DOCUMENTED
- **Issue**: Lines 80, 86: `.into_scalar()` in iterative peak-finding loop
- **Impact**: Called `num_fingers` times (typically 3-5)
- **Solution**: Documented with TODO for GPU topk operation
- **Future**: Replace with `Tensor::topk()` when Burn adds it

---

## ⚠️ Remaining Unavoidable Sync Points

### 1. **modulation.rs - Phase Extraction** (Lines 248-249)
```rust
let real_val: f32 = real_corr.into_scalar().elem();
let imag_val: f32 = imag_corr.into_scalar().elem();
let angle = (imag_val as f64).atan2(real_val as f64);
```
- **Why**: Rust std `atan2()` is CPU-only
- **Frequency**: Once per symbol (hundreds of times per message)
- **Impact**: HIGH - major bottleneck
- **Future Fix**: Implement GPU atan2 kernel or approximate with polynomial

### 2. **rake.rs - Peak Finding Loop**
```rust
let max_val: f32 = remaining_corr.clone().max().into_scalar().elem();
let argmax_val: i64 = remaining_corr.clone().argmax(0).into_scalar().elem();
```
- **Why**: Iterative peak suppression requires conditional logic
- **Frequency**: `num_fingers` iterations (3-5 per message)
- **Impact**: MEDIUM
- **Future Fix**: Implement `topk()` or use custom WGPU kernel

### 3. **wav.rs - File Output**
```rust
let data = signal.clone().into_data();
```
- **Why**: File I/O requires CPU memory
- **Frequency**: Once per message (final output)
- **Impact**: LOW - acceptable for final output

---

## GPU Pipeline Optimization Strategy

### Phase 1: ✅ Completed
1. ✅ Create GPU-native test utilities
2. ✅ Add GPU-only versions of key functions
3. ✅ Document all sync points with ⚠️ warnings

### Phase 2: 🚧 In Progress
4. **Eliminate atan2 bottleneck** (modulation.rs)
   - Option A: WGPU compute shader for atan2
   - Option B: Polynomial approximation on GPU
   - Option C: Process symbols in batches, sync once

5. **Optimize RAKE peak finding** (rake.rs)
   - Wait for Burn `topk()` API
   - Or: Custom WGPU kernel for multi-peak detection

### Phase 3: Future
6. **Full GPU Decoder Pipeline**
   ```
   read_wav() → [GPU Stream] → write_wav()
        ↓
   synchronize_signal_gpu()
        ↓
   demodulate_symbols_gpu() ← NEW
        ↓
   polar_decode_gpu()
        ↓
   prepare_wav_signal_gpu()
        ↓
   [SYNC] write_wav()
   ```

---

## Performance Metrics (TODO)

| **Operation** | **Before** | **After** | **Improvement** |
|---------------|------------|-----------|-----------------|
| Synchronization | ? ms | ? ms | TBD |
| Demodulation | ? ms | ? ms | TBD |
| RAKE Combining | ? ms | ? ms | TBD |
| **Total Decode** | ? ms | ? ms | TBD |

---

## Developer Guidelines

### ✅ DO:
- Use `*_gpu()` variants in hot paths
- Defer `.into_scalar()` until absolutely necessary
- Batch operations to minimize sync calls
- Document sync points with `⚠️ **SYNC POINT**` comments

### ❌ DON'T:
- Call `.to_data()` in loops
- Use `println!("{:?}", tensor)` in hot paths
- Extract scalars for intermediate calculations

### 🔍 Finding Sync Points:
```bash
cd /home/rustuser/projects/rust/burnApps/bachmodem
rg "\.to_data\(|\.into_data\(|\.into_scalar\(|println!\(.*Tensor" src/
```

---

## References
- [How Burn Fuses Tensor Ops on the GPU](https://www.youtube.com/watch?v=dSRbz9w-HK4)
- Burn Book: [Backend Architecture](https://burn.dev/book/building-blocks/backend.html)
