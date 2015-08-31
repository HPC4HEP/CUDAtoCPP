# CUDAtoCPP

Goal: translating source-to-source CUDA to C++

Using clang's ASTs we translate the CUDA source code to C++.

To run the tool:

1- expand.sh progam.cu > temp.cu
2- translate.sh temp.cu > program.cpp
