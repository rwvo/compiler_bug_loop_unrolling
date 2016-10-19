# Compiler issues: full loop unrolling leads to too many registers used
This code demonstrates a compiler issue with hcc-lc from the ROCm 1.2 distribution.

## Building and running
To compile with hcc-lc: `make clean; make isa; ./matmul`. This version demonstrates the bug: it hangs.

To compile with hcc-hsail: `make clean; make hsail; ./matmul`. This version behaves correctly: it returns almost 
instantaneously.

## Testing environment
I compiled and ran all code examples on an i7 6700k with an AMD R9 Nano (Fiji) dGPU, with ROMc 1.2.

## Code description
Kernel does a tiled version of matrix muliplication: each workgroup loads a tile from input matrices A and B into tile 
static memory. Each work item loads a single value of the tiles for A and B. In the outer loop, the input tile for A shifts
to the right in each iteration, and the input tile for B shifts down. In the inner loop, each work item computes a partial
dot product using values from tile static memory; when the outer loop finishes, all dot products have been computed, and they
are stored into the output matrix C.

The program computes the matrix product of two matrices with uninitialized values; the resulting matrix will therefore contain
garbage. No ouput is printed; the program just returns after the matrix multiplication.

## Expected behavior
The program should finish (quickly), without printing any output.

## Actual behavior
When running the program compiled with hcc-hsail, the program behaves as expected. When running the program compiled with hcc-lc,
the program hangs.

## Generated assembly
I dumped the assembly for both the version compiled with hcc-lc and hcc-hsail using rocm-gdb, and included them in the repository.
In the hcc-lc version, we see that the workitem_vgpr_count is 137. This is for a tile size of 32 * 32. It follows that the total
register size for a workgroup is 32 * 32 * 137 * 4, which exceeds the size of the register file for a single GCN 3.0 
Compute Unit (4 * 64Kb).

Cause of the high workitem_vgpr_count: the full unrolling of the inner loop of the kernel (32 iterations, using two registers
per iteration for reading from tile static memory). All the reads from tile static memory are done first; then, all the fma 
operations are done.

The hcc-hsail version does not do full unrolling; the unroll factor is only two, leading to a much smaller amount of registers
used, but also to loss of performance.

## Remarks
From performance experiments, it follows that loop unrolling has a considerable impact on performance, and I would prefer not
to give that up. I see two obvious ways to keep (some) unrolling, but using fewer registers:

Current code (schematically)
```
ds_read_b32 v1, ... // read all 32 x 2 values
ds_read_b32 v2, ...
...
ds_read_b32 v64, ...
s_waitcnt     lgkmcnt(0)
v_mac_f32 v0, v1, v2 // do all 32 fma ops
v_mac_f32 v0, v3, v4
...
v_mac_f32 v0, v63, v64
```

Alternative 1: do partial unrolling. E.g., original loop has 32 iterations; unroll 8 iterations, and loop 4 times.

Alternative 2: fully unroll loop, but interleave reads and fma ops, so that registers can be reused:
```
ds_read_b32 v1, ... // read first 8 x 2 values
ds_read_b32 v2, ...
...
ds_read_b32 v16, ...
s_waitcnt     lgkmcnt(0)
v_mac_f32 v0, v1, v2 // do first 8 fma ops
v_mac_f32 v0, v3, v4
...
v_mac_f32 v0, v15, v16 

ds_read_b32 v1, ... // read next 8 x 2 values
ds_read_b32 v2, ...
...
ds_read_b32 v16, ...
s_waitcnt     lgkmcnt(0)
v_mac_f32 v0, v1, v2 // do next 8 fma ops
v_mac_f32 v0, v3, v4
...
v_mac_f32 v0, v15, v16
// repeat twice more
```

Remark: the alternative code, like the original code, does (several) `s_waitcnt lgkmcnt(0)`. In other words: the fma ops only 
start when *all* the preceding loads have completed. There seems to be room for improvement by overlapping data transfer and 
computation here, e.g,

Alternative 3:
```
ds_read_b32 v1, ... // read first 8 x 2 values 
ds_read_b32 v2, ...
...
ds_read_b32 v16, ... // 16 reads scheduled until here

s_waitcnt lgkmcnt(14) // wait *only* for first two values to become availabe
v_mac_f32 v0, v1, v2  // use them in computation
ds_read_b32 v1, ...   // immediately schedule subsequent read instructions reusing the 
ds_read_b32 v2, ...   // first two registers, so again, 16 outstanding read instructions
 
s_waitcnt lgkmcnt(14) // wait *only* for next two values to become availabe
v_mac_f32 v0, v3, v4  // and so on
ds_read_b32 v3, ...   
ds_read_b32 v4, ...   
 
s_waitcnt lgkmcnt(14) // and so on, and so forth
v_mac_f32 v0, v5, v6  
ds_read_b32 v5, ...   
ds_read_b32 v6, ...

...
```

If I understand it correctly, s_waitcnt doesn't use a functional unit, i.e., it is more or less free, so using more of them 
in alternative 3 (compared to alternative 2) doesn't come with a penalty, while I would expect the latency in alternative 3 
to be reduced.




