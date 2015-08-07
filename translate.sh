#!/bin/bash
#BIN=build/CUDARewriter
BIN=build/TopLevel

##It doesn't seem to work including the cuda header... clang generates the ast, but our tool doesn't :(
#HEADER=src/cuda.h
#HEADER=src/cuda2.h
HEADER=../include/cuda_runtime.h

EXTRA_INCLUDES="-I /usr/local/lib/clang/3.7.0/include"

$BIN $@ -- -Xclang -fsyntax-only -D __CUDACC__ -D __SM_35_INTRINSICS_H__ -D __SURFACE_INDIRECT_FUNCTIONS_H__ -D __SM_32_INTRINSICS_H__ -include $HEADER $EXTRA_INCLUDES -v
