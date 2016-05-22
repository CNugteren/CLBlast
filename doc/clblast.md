CLBlast: API reference
================


xSWAP: Swap two vectors
-------------

Interchanges the contents of vectors x and y.

C++ API:
```
template <typename T>
StatusCode Swap(const size_t n,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SWAP:

* `const size_t n`: Integer size argument.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSCAL: Vector scaling
-------------

Multiplies all elements of vector x by a scalar constant alpha.

C++ API:
```
template <typename T>
StatusCode Scal(const size_t n,
                const T alpha,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSscal(const size_t n,
                        const float alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDscal(const size_t n,
                        const double alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCscal(const size_t n,
                        const cl_float2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZscal(const size_t n,
                        const cl_double2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHscal(const size_t n,
                        const cl_half alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SCAL:

* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xCOPY: Vector copy
-------------

Copies the contents of vector x into vector y.

C++ API:
```
template <typename T>
StatusCode Copy(const size_t n,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastScopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to COPY:

* `const size_t n`: Integer size argument.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAXPY: Vector-times-constant plus vector
-------------

Performs the operation y = alpha * x + y, in which x and y are vectors and alpha is a scalar constant.

C++ API:
```
template <typename T>
StatusCode Axpy(const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSaxpy(const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDaxpy(const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCaxpy(const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZaxpy(const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHaxpy(const size_t n,
                        const cl_half alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to AXPY:

* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xDOT: Dot product of two vectors
-------------

Multiplies the vectors x and y element-wise and accumulates the results. The sum is stored in the dot buffer.

C++ API:
```
template <typename T>
StatusCode Dot(const size_t n,
               cl_mem dot_buffer, const size_t dot_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to DOT:

* `const size_t n`: Integer size argument.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xDOTU: Dot product of two complex vectors
-------------

See the regular xDOT routine.

C++ API:
```
template <typename T>
StatusCode Dotu(const size_t n,
                cl_mem dot_buffer, const size_t dot_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to DOTU:

* `const size_t n`: Integer size argument.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xDOTC: Dot product of two complex vectors, one conjugated
-------------

See the regular xDOT routine.

C++ API:
```
template <typename T>
StatusCode Dotc(const size_t n,
                cl_mem dot_buffer, const size_t dot_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to DOTC:

* `const size_t n`: Integer size argument.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xNRM2: Euclidian norm of a vector
-------------

Accumulates the square of each element in the x vector and takes the square root. The resulting L2 norm is stored in the nrm2 buffer.

C++ API:
```
template <typename T>
StatusCode Nrm2(const size_t n,
                cl_mem nrm2_buffer, const size_t nrm2_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastScnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDznrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to NRM2:

* `const size_t n`: Integer size argument.
* `cl_mem nrm2_buffer`: OpenCL buffer to store the output nrm2 vector.
* `const size_t nrm2_offset`: The offset in elements from the start of the output nrm2 vector.
* `cl_mem nrm2_buffer`: OpenCL buffer to store the output nrm2 vector.
* `const size_t nrm2_offset`: The offset in elements from the start of the output nrm2 vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xASUM: Absolute sum of values in a vector
-------------

Accumulates the absolute value of each element in the x vector. The results are stored in the asum buffer.

C++ API:
```
template <typename T>
StatusCode Asum(const size_t n,
                cl_mem asum_buffer, const size_t asum_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastScasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDzasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to ASUM:

* `const size_t n`: Integer size argument.
* `cl_mem asum_buffer`: OpenCL buffer to store the output asum vector.
* `const size_t asum_offset`: The offset in elements from the start of the output asum vector.
* `cl_mem asum_buffer`: OpenCL buffer to store the output asum vector.
* `const size_t asum_offset`: The offset in elements from the start of the output asum vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSUM: Sum of values in a vector (non-BLAS function)
-------------

Accumulates the values of each element in the x vector. The results are stored in the sum buffer. This routine is the non-absolute version of the xASUM BLAS routine.

C++ API:
```
template <typename T>
StatusCode Sum(const size_t n,
               cl_mem sum_buffer, const size_t sum_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastScsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDzsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastHsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to SUM:

* `const size_t n`: Integer size argument.
* `cl_mem sum_buffer`: OpenCL buffer to store the output sum vector.
* `const size_t sum_offset`: The offset in elements from the start of the output sum vector.
* `cl_mem sum_buffer`: OpenCL buffer to store the output sum vector.
* `const size_t sum_offset`: The offset in elements from the start of the output sum vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAMAX: Index of absolute maximum value in a vector
-------------

Finds the index of the maximum of the absolute values in the x vector. The resulting integer index is stored in the imax buffer.

C++ API:
```
template <typename T>
StatusCode Amax(const size_t n,
                cl_mem imax_buffer, const size_t imax_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastiSamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiDamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiCamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiZamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiHamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to AMAX:

* `const size_t n`: Integer size argument.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xMAX: Index of maximum value in a vector (non-BLAS function)
-------------

Finds the index of the maximum of the values in the x vector. The resulting integer index is stored in the imax buffer. This routine is the non-absolute version of the IxAMAX BLAS routine.

C++ API:
```
template <typename T>
StatusCode Max(const size_t n,
               cl_mem imax_buffer, const size_t imax_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastiSmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiDmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiCmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiZmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiHmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to MAX:

* `const size_t n`: Integer size argument.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xMIN: Index of minimum value in a vector (non-BLAS function)
-------------

Finds the index of the minimum of the values in the x vector. The resulting integer index is stored in the imin buffer. This routine is the non-absolute minimum version of the IxAMAX BLAS routine.

C++ API:
```
template <typename T>
StatusCode Min(const size_t n,
               cl_mem imin_buffer, const size_t imin_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastiSmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiDmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiCmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiZmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastiHmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to MIN:

* `const size_t n`: Integer size argument.
* `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
* `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
* `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
* `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGEMV: General matrix-vector multiplication
-------------

Performs the operation y = alpha * A * x + beta * y, in which x is an input vector, y is an input and output vector, A is an input matrix, and alpha and beta are scalars. The matrix A can optionally be transposed before performing the operation.

C++ API:
```
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to GEMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGBMV: General banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is banded instead.

C++ API:
```
template <typename T>
StatusCode Gbmv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n, const size_t kl, const size_t ku,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to GBMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const size_t kl`: Integer size argument.
* `const size_t ku`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHEMV: Hermitian matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is an Hermitian matrix instead.

C++ API:
```
template <typename T>
StatusCode Hemv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HEMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHBMV: Hermitian banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is an Hermitian banded matrix instead.

C++ API:
```
template <typename T>
StatusCode Hbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HBMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHPMV: Hermitian packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is an Hermitian packed matrix instead and represented as AP.

C++ API:
```
template <typename T>
StatusCode Hpmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem ap_buffer, const size_t ap_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChpmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhpmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HPMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYMV: Symmetric matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is symmetric instead.

C++ API:
```
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SYMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSBMV: Symmetric banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is symmetric and banded instead.

C++ API:
```
template <typename T>
StatusCode Sbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SBMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSPMV: Symmetric packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is a symmetric packed matrix instead and represented as AP.

C++ API:
```
template <typename T>
StatusCode Spmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem ap_buffer, const size_t ap_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SPMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTRMV: Triangular matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is triangular instead.

C++ API:
```
template <typename T>
StatusCode Trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastStrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to TRMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for a non-unit values on the diagonal or `Diagonal::kUnit` (132) for a unit values on the diagonal.
* `const size_t n`: Integer size argument.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTBMV: Triangular banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is triangular and banded instead.

C++ API:
```
template <typename T>
StatusCode Tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n, const size_t k,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastStbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to TBMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for a non-unit values on the diagonal or `Diagonal::kUnit` (132) for a unit values on the diagonal.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTPMV: Triangular packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix A is a triangular packed matrix instead and repreented as AP.

C++ API:
```
template <typename T>
StatusCode Tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem ap_buffer, const size_t ap_offset,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastStpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to TPMV:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for a non-unit values on the diagonal or `Diagonal::kUnit` (132) for a unit values on the diagonal.
* `const size_t n`: Integer size argument.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGER: General rank-1 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Ger(const Layout layout,
               const size_t m, const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSger(const Layout layout,
                       const size_t m, const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDger(const Layout layout,
                       const size_t m, const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to GER:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGERU: General rank-1 complex matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Geru(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to GERU:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGERC: General rank-1 complex conjugated matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Gerc(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to GERC:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHER: Hermitian rank-1 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Her(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to HER:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHPR: Hermitian packed rank-1 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Hpr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem ap_buffer, const size_t ap_offset,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to HPR:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHER2: Hermitian rank-2 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Her2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HER2:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHPR2: Hermitian packed rank-2 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Hpr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem ap_buffer, const size_t ap_offset,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HPR2:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYR: Symmetric rank-1 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Syr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to SYR:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSPR: Symmetric packed rank-1 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Spr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem ap_buffer, const size_t ap_offset,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event)
```

Arguments to SPR:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYR2: Symmetric rank-2 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Syr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SYR2:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSPR2: Symmetric packed rank-2 matrix update
-------------



C++ API:
```
template <typename T>
StatusCode Spr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem ap_buffer, const size_t ap_offset,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SPR2:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGEMM: General matrix-matrix multiplication
-------------



C++ API:
```
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                const size_t m, const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to GEMM:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYMM: Symmetric matrix-matrix multiplication
-------------



C++ API:
```
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SYMM:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side`: The horizontal position of the triangular matrix, either `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHEMM: Hermitian matrix-matrix multiplication
-------------



C++ API:
```
template <typename T>
StatusCode Hemm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastChemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HEMM:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side`: The horizontal position of the triangular matrix, either `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYRK: Rank-K update of a symmetric matrix
-------------



C++ API:
```
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to SYRK:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHERK: Rank-K update of a hermitian matrix
-------------



C++ API:
```
template <typename T>
StatusCode Herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to HERK:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYR2K: Rank-2K update of a symmetric matrix
-------------



C++ API:
```
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                 const T beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastSsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
```

Arguments to SYR2K:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHER2K: Rank-2K update of a hermitian matrix
-------------



C++ API:
```
template <typename T, typename U>
StatusCode Her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                 const U beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastCher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event)
```

Arguments to HER2K:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument.
* `const size_t k`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix.
* `const U beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTRMM: Triangular matrix-matrix multiplication
-------------



C++ API:
```
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
StatusCode CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event)
StatusCode CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event)
```

Arguments to TRMM:

* `const Layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side`: The horizontal position of the triangular matrix, either `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle`: The vertical position of the triangular matrix, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for a non-unit values on the diagonal or `Diagonal::kUnit` (132) for a unit values on the diagonal.
* `const size_t m`: Integer size argument.
* `const size_t n`: Integer size argument.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix.
* `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
* `const size_t b_offset`: The offset in elements from the start of the output B matrix.
* `const size_t b_ld`: Leading dimension of the output B matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



