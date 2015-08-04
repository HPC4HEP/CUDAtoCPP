#!/bin/bash
BIN=build/CUDARewriter
#BIN=build/TopLevel

##It doesn't seem to work including the cuda header... clang generates the ast, but our tool doesn't :(
HEADER=src/cuda2.h
#HEADER=../../include/cuda_runtime.h

$BIN $@ -- -Xclang -fsyntax-only -D __CUDACC__ -include $HEADER
