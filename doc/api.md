CLBlast: API reference
================


xSWAP: Swap two vectors
-------------

Interchanges _n_ elements of vectors _x_ and _y_.

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
CLBlastStatusCode CLBlastSswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SWAP:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSCAL: Vector scaling
-------------

Multiplies _n_ elements of vector _x_ by a scalar constant _alpha_.

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
CLBlastStatusCode CLBlastSscal(const size_t n,
                               const float alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDscal(const size_t n,
                               const double alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCscal(const size_t n,
                               const cl_float2 alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZscal(const size_t n,
                               const cl_double2 alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHscal(const size_t n,
                               const cl_half alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SCAL:

* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xCOPY: Vector copy
-------------

Copies the contents of vector _x_ into vector _y_.

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
CLBlastStatusCode CLBlastScopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to COPY:

* `const size_t n`: Integer size argument. This value must be positive.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAXPY: Vector-times-constant plus vector
-------------

Performs the operation _y = alpha * x + y_, in which _x_ and _y_ are vectors and _alpha_ is a scalar constant.

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
CLBlastStatusCode CLBlastSaxpy(const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDaxpy(const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCaxpy(const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZaxpy(const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHaxpy(const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to AXPY:

* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xDOT: Dot product of two vectors
-------------

Multiplies _n_ elements of the vectors _x_ and _y_ element-wise and accumulates the results. The sum is stored in the _dot_ buffer.

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
CLBlastStatusCode CLBlastSdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to DOT:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
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
CLBlastStatusCode CLBlastCdotu(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZdotu(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to DOTU:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
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
CLBlastStatusCode CLBlastCdotc(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZdotc(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to DOTC:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
* `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xNRM2: Euclidian norm of a vector
-------------

Accumulates the square of _n_ elements in the _x_ vector and takes the square root. The resulting L2 norm is stored in the _nrm2_ buffer.

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
CLBlastStatusCode CLBlastSnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastScnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDznrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to NRM2:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem nrm2_buffer`: OpenCL buffer to store the output nrm2 vector.
* `const size_t nrm2_offset`: The offset in elements from the start of the output nrm2 vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xASUM: Absolute sum of values in a vector
-------------

Accumulates the absolute value of _n_ elements in the _x_ vector. The results are stored in the _asum_ buffer.

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
CLBlastStatusCode CLBlastSasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastScasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDzasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to ASUM:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem asum_buffer`: OpenCL buffer to store the output asum vector.
* `const size_t asum_offset`: The offset in elements from the start of the output asum vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSUM: Sum of values in a vector (non-BLAS function)
-------------

Accumulates the values of _n_ elements in the _x_ vector. The results are stored in the _sum_ buffer. This routine is the non-absolute version of the xASUM BLAS routine.

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
CLBlastStatusCode CLBlastSsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastScsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDzsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to SUM:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem sum_buffer`: OpenCL buffer to store the output sum vector.
* `const size_t sum_offset`: The offset in elements from the start of the output sum vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAMAX: Index of absolute maximum value in a vector
-------------

Finds the index of the maximum of the absolute values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer.

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
CLBlastStatusCode CLBlastiSamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiDamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiCamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiZamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiHamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to AMAX:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAMIN: Index of absolute minimum value in a vector (non-BLAS function)
-------------

Finds the index of the minimum of the absolute values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer.

C++ API:
```
template <typename T>
StatusCode Amin(const size_t n,
                cl_mem imin_buffer, const size_t imin_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastiSamin(const size_t n,
                               cl_mem imin_buffer, const size_t imin_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiDamin(const size_t n,
                               cl_mem imin_buffer, const size_t imin_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiCamin(const size_t n,
                               cl_mem imin_buffer, const size_t imin_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiZamin(const size_t n,
                               cl_mem imin_buffer, const size_t imin_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiHamin(const size_t n,
                               cl_mem imin_buffer, const size_t imin_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to AMIN:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
* `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xMAX: Index of maximum value in a vector (non-BLAS function)
-------------

Finds the index of the maximum of the values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer. This routine is the non-absolute version of the IxAMAX BLAS routine.

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
CLBlastStatusCode CLBlastiSmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiDmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiCmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiZmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiHmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to MAX:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
* `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xMIN: Index of minimum value in a vector (non-BLAS function)
-------------

Finds the index of the minimum of the values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer. This routine is the non-absolute minimum version of the IxAMAX BLAS routine.

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
CLBlastStatusCode CLBlastiSmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiDmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiCmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiZmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastiHmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to MIN:

* `const size_t n`: Integer size argument. This value must be positive.
* `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
* `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGEMV: General matrix-vector multiplication
-------------

Performs the operation _y = alpha * A * x + beta * y_, in which _x_ is an input vector, _y_ is an input and output vector, _A_ is an input matrix, and _alpha_ and _beta_ are scalars. The matrix _A_ can optionally be transposed before performing the operation.

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
CLBlastStatusCode CLBlastSgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to GEMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GEMV:

* The value of `a_ld` must be at least `m`.



xGBMV: General banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is banded instead.

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
CLBlastStatusCode CLBlastSgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to GBMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t kl`: Integer size argument. This value must be positive.
* `const size_t ku`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GBMV:

* The value of `a_ld` must be at least `kl + ku + 1`.



xHEMV: Hermitian matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is an Hermitian matrix instead.

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
CLBlastStatusCode CLBlastChemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HEMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HEMV:

* The value of `a_ld` must be at least `n`.



xHBMV: Hermitian banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is an Hermitian banded matrix instead.

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
CLBlastStatusCode CLBlastChbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HBMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HBMV:

* The value of `a_ld` must be at least `k + 1`.



xHPMV: Hermitian packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastChpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HPMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYMV: Symmetric matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is symmetric instead.

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
CLBlastStatusCode CLBlastSsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SYMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYMV:

* The value of `a_ld` must be at least `n`.



xSBMV: Symmetric banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is symmetric and banded instead.

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
CLBlastStatusCode CLBlastSsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SBMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SBMV:

* The value of `a_ld` must be at least `k + 1`.



xSPMV: Symmetric packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastSspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SPMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t y_offset`: The offset in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTRMV: Triangular matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is triangular instead.

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
CLBlastStatusCode CLBlastStrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TRMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t n`: Integer size argument. This value must be positive.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for TRMV:

* The value of `a_ld` must be at least `n`.



xTBMV: Triangular banded matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is triangular and banded instead.

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
CLBlastStatusCode CLBlastStbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TBMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for TBMV:

* The value of `a_ld` must be at least `k + 1`.



xTPMV: Triangular packed matrix-vector multiplication
-------------

Same operation as xGEMV, but matrix _A_ is a triangular packed matrix instead and repreented as _AP_.

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
CLBlastStatusCode CLBlastStpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TPMV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t n`: Integer size argument. This value must be positive.
* `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xTRSV: Solves a triangular system of equations
-------------



C++ API:
```
template <typename T>
StatusCode Trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastStrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TRSV:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t n`: Integer size argument. This value must be positive.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
* `const size_t x_offset`: The offset in elements from the start of the output x vector.
* `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGER: General rank-1 matrix update
-------------

Performs the operation _A = alpha * x * y^T + A_, in which _x_ is an input vector, _y^T_ is the transpose of the input vector _y_, _A_ is the matrix to be updated, and _alpha_ is a scalar value.

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
CLBlastStatusCode CLBlastSger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to GER:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GER:

* The value of `a_ld` must be at least `m`.



xGERU: General rank-1 complex matrix update
-------------

Same operation as xGER, but with complex data-types.

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
CLBlastStatusCode CLBlastCgeru(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgeru(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to GERU:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GERU:

* The value of `a_ld` must be at least `m`.



xGERC: General rank-1 complex conjugated matrix update
-------------

Same operation as xGERU, but the update is done based on the complex conjugate of the input vectors.

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
CLBlastStatusCode CLBlastCgerc(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgerc(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to GERC:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GERC:

* The value of `a_ld` must be at least `m`.



xHER: Hermitian rank-1 matrix update
-------------

Performs the operation _A = alpha * x * x^T + A_, in which x is an input vector, x^T is the transpose of this vector, _A_ is the triangular Hermetian matrix to be updated, and alpha is a scalar value.

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
CLBlastStatusCode CLBlastCher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to HER:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HER:

* The value of `a_ld` must be at least `n`.



xHPR: Hermitian packed rank-1 matrix update
-------------

Same operation as xHER, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastChpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to HPR:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHER2: Hermitian rank-2 matrix update
-------------

Performs the operation _A = alpha * x * y^T + conj(alpha) * y * x^T + A_, in which _x_ is an input vector and _x^T_ its transpose, _y_ is an input vector and _y^T_ its transpose, _A_ is the triangular Hermetian matrix to be updated, _alpha_ is a scalar value and _conj(alpha)_ its complex conjugate.

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
CLBlastStatusCode CLBlastCher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HER2:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HER2:

* The value of `a_ld` must be at least `n`.



xHPR2: Hermitian packed rank-2 matrix update
-------------

Same operation as xHER2, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastChpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HPR2:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYR: Symmetric rank-1 matrix update
-------------

Same operation as xHER, but matrix A is a symmetric matrix instead.

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
CLBlastStatusCode CLBlastSsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to SYR:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYR:

* The value of `a_ld` must be at least `n`.



xSPR: Symmetric packed rank-1 matrix update
-------------

Same operation as xSPR, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastSspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to SPR:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xSYR2: Symmetric rank-2 matrix update
-------------

Same operation as xHER2, but matrix _A_ is a symmetric matrix instead.

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
CLBlastStatusCode CLBlastSsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SYR2:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
* `const size_t a_offset`: The offset in elements from the start of the output A matrix.
* `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYR2:

* The value of `a_ld` must be at least `n`.



xSPR2: Symmetric packed rank-2 matrix update
-------------

Same operation as xSPR2, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.

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
CLBlastStatusCode CLBlastSspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SPR2:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
* `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGEMM: General matrix-matrix multiplication
-------------

Performs the matrix product _C = alpha * A * B + beta * C_, in which _A_ (_m_ by _k_) and _B_ (_k_ by _n_) are two general rectangular input matrices, _C_ (_m_ by _n_) is the matrix to be updated, and _alpha_ and _beta_ are scalar values. The matrices _A_ and/or _B_ can optionally be transposed before performing the operation.

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
                cl_command_queue* queue, cl_event* event,
                cl_mem temp_buffer = nullptr)
```

C API:
```
CLBlastStatusCode CLBlastSgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to GEMM:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GEMM:

* When `transpose_a == Transpose::kNo`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
* When `transpose_b == Transpose::kNo`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
* The value of `c_ld` must be at least `m`.



xSYMM: Symmetric matrix-matrix multiplication
-------------

Same operation as xGEMM, but _A_ is symmetric instead. In case of `side == kLeft`, _A_ is a symmetric _m_ by _m_ matrix and _C = alpha * A * B + beta * C_ is performed. Otherwise, in case of `side == kRight`, _A_ is a symmtric _n_ by _n_ matrix and _C = alpha * B * A + beta * C_ is performed.

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
CLBlastStatusCode CLBlastSsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SYMM:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYMM:

* When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
* The value of `b_ld` must be at least `m`.
* The value of `c_ld` must be at least `m`.



xHEMM: Hermitian matrix-matrix multiplication
-------------

Same operation as xSYMM, but _A_ is an Hermitian matrix instead.

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
CLBlastStatusCode CLBlastChemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HEMM:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HEMM:

* When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
* The value of `b_ld` must be at least `m`.
* The value of `c_ld` must be at least `m`.



xSYRK: Rank-K update of a symmetric matrix
-------------

Performs the matrix product _C = alpha * A * A^T + beta * C_ or _C = alpha * A^T * A + beta * C_, in which _A_ is a general matrix and _A^T_ is its transpose, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.

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
CLBlastStatusCode CLBlastSsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to SYRK:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYRK:

* When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
* The value of `c_ld` must be at least `m`.



xHERK: Rank-K update of a hermitian matrix
-------------

Same operation as xSYRK, but _C_ is an Hermitian matrix instead.

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
CLBlastStatusCode CLBlastCherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to HERK:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HERK:

* When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
* The value of `c_ld` must be at least `m`.



xSYR2K: Rank-2K update of a symmetric matrix
-------------

Performs the matrix product _C = alpha * A * B^T + alpha * B * A^T + beta * C_ or _C = alpha * A^T * B + alpha * B^T * A + beta * C_, in which _A_ and _B_ are general matrices and _A^T_ and _B^T_ are their transposed versions, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.

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
CLBlastStatusCode CLBlastSsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const float alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const float beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const double alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const double beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_float2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_float2 beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_double2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_double2 beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_half alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_half beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
```

Arguments to SYR2K:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose ab_transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for SYR2K:

* When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
* When `transpose == Transpose::kNo`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`.
* The value of `c_ld` must be at least `n`.



xHER2K: Rank-2K update of a hermitian matrix
-------------

Same operation as xSYR2K, but _C_ is an Hermitian matrix instead.

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
CLBlastStatusCode CLBlastCher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_float2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const float beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_double2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const double beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event)
```

Arguments to HER2K:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose ab_transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const U beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for HER2K:

* When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
* When `transpose == Transpose::kNo`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`.
* The value of `c_ld` must be at least `n`.



xTRMM: Triangular matrix-matrix multiplication
-------------

Performs the matrix product _B = alpha * A * B_ or _B = alpha * B * A_, in which _A_ is a unit or non-unit triangular matrix, _B_ (_m_ by _n_) is the general matrix to be updated, and _alpha_ is a scalar value.

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
CLBlastStatusCode CLBlastStrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TRMM:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
* `const size_t b_offset`: The offset in elements from the start of the output B matrix.
* `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for TRMM:

* When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
* The value of `b_ld` must be at least `m`.



xTRSM: Solves a triangular system of equations
-------------

Solves the equation _A * X = alpha * B_ for the unknown _m_ by _n_ matrix X, in which _A_ is an _n_ by _n_ unit or non-unit triangular matrix and B is an _m_ by _n_ matrix. The matrix _B_ is overwritten by the solution _X_.

C++ API:
```
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastStrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event)
```

Arguments to TRSM:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
* `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
* `const size_t b_offset`: The offset in elements from the start of the output B matrix.
* `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xHAD: Element-wise vector product (Hadamard)
-------------

Performs the Hadamard element-wise product _z = alpha * x * y + beta * z_, in which _x_, _y_, and _z_ are vectors and _alpha_ and _beta_ are scalar constants.

C++ API:
```
template <typename T>
StatusCode Had(const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               const T beta,
               cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
               cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastShad(const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const float beta,
                              cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDhad(const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const double beta,
                              cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastChad(const size_t n,
                              const cl_float2 alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const cl_float2 beta,
                              cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZhad(const size_t n,
                              const cl_double2 alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const cl_double2 beta,
                              cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                              cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHhad(const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const cl_half beta,
                              cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                              cl_command_queue* queue, cl_event* event)
```

Arguments to HAD:

* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t x_offset`: The offset in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
* `const size_t y_offset`: The offset in elements from the start of the input y vector.
* `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
* `const T beta`: Input scalar constant.
* `cl_mem z_buffer`: OpenCL buffer to store the output z vector.
* `const size_t z_offset`: The offset in elements from the start of the output z vector.
* `const size_t z_inc`: Stride/increment of the output z vector. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xOMATCOPY: Scaling and out-place transpose/copy (non-BLAS function)
-------------

Performs scaling and out-of-place transposition/copying of matrices according to _B = alpha*op(A)_, in which _A_ is an input matrix (_m_ rows by _n_ columns), _B_ an output matrix, and _alpha_ a scalar value. The operation _op_ can be a normal matrix copy, a transposition or a conjugate transposition.

C++ API:
```
template <typename T>
StatusCode Omatcopy(const Layout layout, const Transpose a_transpose,
                    const size_t m, const size_t n,
                    const T alpha,
                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                    cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                    cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastComatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_half alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event)
```

Arguments to OMATCOPY:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
* `const size_t b_offset`: The offset in elements from the start of the output B matrix.
* `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for OMATCOPY:

* The value of `a_ld` must be at least `m`.
* The value of `b_ld` must be at least `n`.



xIM2COL: Im2col function (non-BLAS function)
-------------

Performs the im2col algorithm, in which _im_ is the input matrix and _col_ is the output matrix. Overwrites any existing values in the _col_ buffer

C++ API:
```
template <typename T>
StatusCode Im2col(const KernelMode kernel_mode,
                  const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                  const cl_mem im_buffer, const size_t im_offset,
                  cl_mem col_buffer, const size_t col_offset,
                  cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSim2col(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem im_buffer, const size_t im_offset,
                                 cl_mem col_buffer, const size_t col_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDim2col(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem im_buffer, const size_t im_offset,
                                 cl_mem col_buffer, const size_t col_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCim2col(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem im_buffer, const size_t im_offset,
                                 cl_mem col_buffer, const size_t col_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZim2col(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem im_buffer, const size_t im_offset,
                                 cl_mem col_buffer, const size_t col_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHim2col(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem im_buffer, const size_t im_offset,
                                 cl_mem col_buffer, const size_t col_offset,
                                 cl_command_queue* queue, cl_event* event)
```

Arguments to IM2COL:

* `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
* `const size_t channels`: Integer size argument. This value must be positive.
* `const size_t height`: Integer size argument. This value must be positive.
* `const size_t width`: Integer size argument. This value must be positive.
* `const size_t kernel_h`: Integer size argument. This value must be positive.
* `const size_t kernel_w`: Integer size argument. This value must be positive.
* `const size_t pad_h`: Integer size argument. This value must be positive.
* `const size_t pad_w`: Integer size argument. This value must be positive.
* `const size_t stride_h`: Integer size argument. This value must be positive.
* `const size_t stride_w`: Integer size argument. This value must be positive.
* `const size_t dilation_h`: Integer size argument. This value must be positive.
* `const size_t dilation_w`: Integer size argument. This value must be positive.
* `const cl_mem im_buffer`: OpenCL buffer to store the input im tensor.
* `const size_t im_offset`: The offset in elements from the start of the input im tensor.
* `cl_mem col_buffer`: OpenCL buffer to store the output col tensor.
* `const size_t col_offset`: The offset in elements from the start of the output col tensor.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xCOL2IM: Col2im function (non-BLAS function)
-------------

Performs the col2im algorithm, in which _col_ is the input matrix and _im_ is the output matrix. Accumulates results on top of the existing values in the _im_ buffer.

C++ API:
```
template <typename T>
StatusCode Col2im(const KernelMode kernel_mode,
                  const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                  const cl_mem col_buffer, const size_t col_offset,
                  cl_mem im_buffer, const size_t im_offset,
                  cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastScol2im(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem col_buffer, const size_t col_offset,
                                 cl_mem im_buffer, const size_t im_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDcol2im(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem col_buffer, const size_t col_offset,
                                 cl_mem im_buffer, const size_t im_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCcol2im(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem col_buffer, const size_t col_offset,
                                 cl_mem im_buffer, const size_t im_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZcol2im(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem col_buffer, const size_t col_offset,
                                 cl_mem im_buffer, const size_t im_offset,
                                 cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHcol2im(const CLBlastKernelMode kernel_mode,
                                 const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                 const cl_mem col_buffer, const size_t col_offset,
                                 cl_mem im_buffer, const size_t im_offset,
                                 cl_command_queue* queue, cl_event* event)
```

Arguments to COL2IM:

* `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
* `const size_t channels`: Integer size argument. This value must be positive.
* `const size_t height`: Integer size argument. This value must be positive.
* `const size_t width`: Integer size argument. This value must be positive.
* `const size_t kernel_h`: Integer size argument. This value must be positive.
* `const size_t kernel_w`: Integer size argument. This value must be positive.
* `const size_t pad_h`: Integer size argument. This value must be positive.
* `const size_t pad_w`: Integer size argument. This value must be positive.
* `const size_t stride_h`: Integer size argument. This value must be positive.
* `const size_t stride_w`: Integer size argument. This value must be positive.
* `const size_t dilation_h`: Integer size argument. This value must be positive.
* `const size_t dilation_w`: Integer size argument. This value must be positive.
* `const cl_mem col_buffer`: OpenCL buffer to store the input col tensor.
* `const size_t col_offset`: The offset in elements from the start of the input col tensor.
* `cl_mem im_buffer`: OpenCL buffer to store the output im tensor.
* `const size_t im_offset`: The offset in elements from the start of the output im tensor.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xCONVGEMM: Batched convolution as GEMM (non-BLAS function)
-------------

Integrates im2col and GEMM for batched 3D convolution, in which _im_ is the 4D input tensor (NCHW - batch-channelin-height-width), _kernel_ the 4D kernel weights tensor (KCHW - channelout-channelin-height-width), and _result_ the 4D output tensor (NCHW - batch-channelout-height-width).

C++ API:
```
template <typename T>
StatusCode Convgemm(const KernelMode kernel_mode,
                    const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                    const cl_mem im_buffer, const size_t im_offset,
                    const cl_mem kernel_buffer, const size_t kernel_offset,
                    cl_mem result_buffer, const size_t result_offset,
                    cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSconvgemm(const CLBlastKernelMode kernel_mode,
                                   const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                   const cl_mem im_buffer, const size_t im_offset,
                                   const cl_mem kernel_buffer, const size_t kernel_offset,
                                   cl_mem result_buffer, const size_t result_offset,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDconvgemm(const CLBlastKernelMode kernel_mode,
                                   const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                   const cl_mem im_buffer, const size_t im_offset,
                                   const cl_mem kernel_buffer, const size_t kernel_offset,
                                   cl_mem result_buffer, const size_t result_offset,
                                   cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHconvgemm(const CLBlastKernelMode kernel_mode,
                                   const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                   const cl_mem im_buffer, const size_t im_offset,
                                   const cl_mem kernel_buffer, const size_t kernel_offset,
                                   cl_mem result_buffer, const size_t result_offset,
                                   cl_command_queue* queue, cl_event* event)
```

Arguments to CONVGEMM:

* `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
* `const size_t channels`: Integer size argument. This value must be positive.
* `const size_t height`: Integer size argument. This value must be positive.
* `const size_t width`: Integer size argument. This value must be positive.
* `const size_t kernel_h`: Integer size argument. This value must be positive.
* `const size_t kernel_w`: Integer size argument. This value must be positive.
* `const size_t pad_h`: Integer size argument. This value must be positive.
* `const size_t pad_w`: Integer size argument. This value must be positive.
* `const size_t stride_h`: Integer size argument. This value must be positive.
* `const size_t stride_w`: Integer size argument. This value must be positive.
* `const size_t dilation_h`: Integer size argument. This value must be positive.
* `const size_t dilation_w`: Integer size argument. This value must be positive.
* `const size_t num_kernels`: Integer size argument. This value must be positive.
* `const size_t batch_count`: Integer size argument. This value must be positive.
* `const cl_mem im_buffer`: OpenCL buffer to store the input im tensor.
* `const size_t im_offset`: The offset in elements from the start of the input im tensor.
* `const cl_mem kernel_buffer`: OpenCL buffer to store the input kernel tensor.
* `const size_t kernel_offset`: The offset in elements from the start of the input kernel tensor.
* `cl_mem result_buffer`: OpenCL buffer to store the output result tensor.
* `const size_t result_offset`: The offset in elements from the start of the output result tensor.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xAXPYBATCHED: Batched version of AXPY
-------------

As AXPY, but multiple operations are batched together for better performance.

C++ API:
```
template <typename T>
StatusCode AxpyBatched(const size_t n,
                       const T *alphas,
                       const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                       cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                       const size_t batch_count,
                       cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSaxpyBatched(const size_t n,
                                      const float *alphas,
                                      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDaxpyBatched(const size_t n,
                                      const double *alphas,
                                      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCaxpyBatched(const size_t n,
                                      const cl_float2 *alphas,
                                      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZaxpyBatched(const size_t n,
                                      const cl_double2 *alphas,
                                      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHaxpyBatched(const size_t n,
                                      const cl_half *alphas,
                                      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
```

Arguments to AXPYBATCHED:

* `const size_t n`: Integer size argument. This value must be positive.
* `const T *alphas`: Input scalar constants.
* `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
* `const size_t *x_offsets`: The offsets in elements from the start of the input x vector.
* `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
* `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
* `const size_t *y_offsets`: The offsets in elements from the start of the output y vector.
* `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
* `const size_t batch_count`: Number of batches. This value must be positive.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.



xGEMMBATCHED: Batched version of GEMM
-------------

As GEMM, but multiple operations are batched together for better performance.

C++ API:
```
template <typename T>
StatusCode GemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                       const size_t m, const size_t n, const size_t k,
                       const T *alphas,
                       const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                       const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                       const T *betas,
                       cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                       const size_t batch_count,
                       cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                      const size_t m, const size_t n, const size_t k,
                                      const float *alphas,
                                      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                      const float *betas,
                                      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                      const size_t m, const size_t n, const size_t k,
                                      const double *alphas,
                                      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                      const double *betas,
                                      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                      const size_t m, const size_t n, const size_t k,
                                      const cl_float2 *alphas,
                                      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                      const cl_float2 *betas,
                                      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                      const size_t m, const size_t n, const size_t k,
                                      const cl_double2 *alphas,
                                      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                      const cl_double2 *betas,
                                      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                      const size_t m, const size_t n, const size_t k,
                                      const cl_half *alphas,
                                      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                      const cl_half *betas,
                                      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                      const size_t batch_count,
                                      cl_command_queue* queue, cl_event* event)
```

Arguments to GEMMBATCHED:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T *alphas`: Input scalar constants.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t *a_offsets`: The offsets in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t *b_offsets`: The offsets in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const T *betas`: Input scalar constants.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t *c_offsets`: The offsets in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `const size_t batch_count`: Number of batches. This value must be positive.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GEMMBATCHED:

* When `transpose_a == Transpose::kNo`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
* When `transpose_b == Transpose::kNo`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
* The value of `c_ld` must be at least `m`.



xGEMMSTRIDEDBATCHED: StridedBatched version of GEMM
-------------

As GEMM, but multiple strided operations are batched together for better performance.

C++ API:
```
template <typename T>
StatusCode GemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                              const size_t m, const size_t n, const size_t k,
                              const T alpha,
                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                              const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                              const T beta,
                              cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                              const size_t batch_count,
                              cl_command_queue* queue, cl_event* event)
```

C API:
```
CLBlastStatusCode CLBlastSgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                             const size_t m, const size_t n, const size_t k,
                                             const float alpha,
                                             const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                             const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                             const float beta,
                                             cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                             const size_t batch_count,
                                             cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastDgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                             const size_t m, const size_t n, const size_t k,
                                             const double alpha,
                                             const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                             const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                             const double beta,
                                             cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                             const size_t batch_count,
                                             cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastCgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                             const size_t m, const size_t n, const size_t k,
                                             const cl_float2 alpha,
                                             const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                             const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                             const cl_float2 beta,
                                             cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                             const size_t batch_count,
                                             cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastZgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                             const size_t m, const size_t n, const size_t k,
                                             const cl_double2 alpha,
                                             const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                             const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                             const cl_double2 beta,
                                             cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                             const size_t batch_count,
                                             cl_command_queue* queue, cl_event* event)
CLBlastStatusCode CLBlastHgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                             const size_t m, const size_t n, const size_t k,
                                             const cl_half alpha,
                                             const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                             const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                             const cl_half beta,
                                             cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                             const size_t batch_count,
                                             cl_command_queue* queue, cl_event* event)
```

Arguments to GEMMSTRIDEDBATCHED:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const T alpha`: Input scalar constant.
* `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const size_t a_stride`: The (fixed) stride between two batches of the A matrix.
* `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const size_t b_stride`: The (fixed) stride between two batches of the B matrix.
* `const T beta`: Input scalar constant.
* `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `const size_t c_stride`: The (fixed) stride between two batches of the C matrix.
* `const size_t batch_count`: Number of batches. This value must be positive.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.

Requirements for GEMMSTRIDEDBATCHED:

* When `transpose_a == Transpose::kNo`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
* When `transpose_b == Transpose::kNo`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
* The value of `c_ld` must be at least `m`.



GemmTempBufferSize: Retrieves the size of the temporary buffer for GEMM (auxiliary function)
-------------

Retrieves the required size of the temporary buffer for the GEMM kernel for specific arguments and for a specific device/platform and tuning parameters. This could be 0 in case no temporary buffer is required. Arguments are similar to those for GEMM.

C++ API:
```
template <typename T>
StatusCode GemmTempBufferSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                              const size_t m, const size_t n, const size_t k,
                              const size_t a_offset, const size_t a_ld,
                              const size_t b_offset, const size_t b_ld,
                              const size_t c_offset, const size_t c_ld,
                              cl_command_queue* queue, size_t& temp_buffer_size)
```

C API:
```
CLBlastStatusCode CLBlastSGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const size_t a_offset, const size_t a_ld,
                               const size_t b_offset, const size_t b_ld,
                               const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, size_t* temp_buffer_size)

CLBlastStatusCode CLBlastDGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const size_t a_offset, const size_t a_ld,
                               const size_t b_offset, const size_t b_ld,
                               const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, size_t* temp_buffer_size)

CLBlastStatusCode CLBlastCGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const size_t a_offset, const size_t a_ld,
                               const size_t b_offset, const size_t b_ld,
                               const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, size_t* temp_buffer_size)

CLBlastStatusCode CLBlastZGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const size_t a_offset, const size_t a_ld,
                               const size_t b_offset, const size_t b_ld,
                               const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, size_t* temp_buffer_size)

CLBlastStatusCode CLBlastHGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const size_t a_offset, const size_t a_ld,
                               const size_t b_offset, const size_t b_ld,
                               const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, size_t* temp_buffer_size)
```
Arguments to GemmTempBufferSize:

* `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
* `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
* `const size_t m`: Integer size argument. This value must be positive.
* `const size_t n`: Integer size argument. This value must be positive.
* `const size_t k`: Integer size argument. This value must be positive.
* `const size_t a_offset`: The offset in elements from the start of the input A matrix.
* `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
* `const size_t b_offset`: The offset in elements from the start of the input B matrix.
* `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
* `const size_t c_offset`: The offset in elements from the start of the output C matrix.
* `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
* `size_t& temp_buffer_size`: The result of this function: the required buffer size.



ClearCache: Resets the cache of compiled binaries (auxiliary function)
-------------

CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on for the same device. This cache can be cleared to free up system memory or it can be useful in case of debugging.

C++ API:
```
StatusCode ClearCache()
```

C API:
```
CLBlastStatusCode CLBlastClearCache()
```



FillCache: Populates the cache of compiled binaries for a specific device (auxiliary function)
-------------

CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on for the same device. This cache is automatically populated whenever a new binary is created. Thus, the first run of a specific kernel could take extra time. For debugging or performance evaluation purposes, it might be useful to populate the cache upfront. This function populates the cache for all kernels in CLBlast for all precisions, but for a specific device only.

C++ API:
```
StatusCode FillCache(const cl_device_id device)
```

C API:
```
CLBlastStatusCode CLBlastFillCache(const cl_device_id device)
```

Arguments to FillCache:

* `const cl_device_id device`: The OpenCL device to fill the cache for.



RetrieveParameters: Retrieves current tuning parameters (auxiliary function)
-------------

This function retrieves current tuning parameters for a specific device-precision-kernel combination. This can be used for debugging or inspection. See [tuning.md](tuning.md) for more details on which kernel names and parameters are valid.

C++ API:
```
StatusCode RetrieveParameters(const cl_device_id device, const std::string &kernel_name,
                              const Precision precision,
                              std::unordered_map<std::string,size_t> &parameters)
```

A C API is not available for this function.

Arguments to RetrieveParameters (C++ version):

* `const cl_device_id device`: The OpenCL device to query the parameters for.
* `const std::string &kernel_name`: The target kernel name. This has to be one of the existing CLBlast kernels (Xaxpy, Xdot, Xgemv, XgemvFast, XgemvFastRot, Xgemv, Xger, Copy, Pad, Transpose, Padtranspose, Xgemm, or XgemmDirect). If this argument is incorrect, this function will return with the `clblast::kInvalidOverrideKernel` status-code.
* `const Precision precision`: The CLBlast precision enum to query the parameters for.
* `std::unordered_map<std::string,size_t> &parameters`: An unordered map of strings to integers. This will be filled with the current tuning parameters for a specific kernel.



OverrideParameters: Override tuning parameters (auxiliary function)
-------------

This function overrides tuning parameters for a specific device-precision-kernel combination. The next time the target routine is called it will be re-compiled and use the new parameters. All further times (until `OverrideParameters` is called again) it will load the kernel from the cache and thus continue to use the new parameters. Note that the first time after calling `OverrideParameters` a performance drop can be observable due to the re-compilation of the kernel. See [tuning.md](tuning.md) for more details on which kernel names and parameters are valid.

C++ API:
```
StatusCode OverrideParameters(const cl_device_id device, const std::string &kernel_name,
                              const Precision precision,
                              const std::unordered_map<std::string,size_t> &parameters)
```

C API:
```
CLBlastStatusCode CLBlastOverrideParameters(const cl_device_id device, const char* kernel_name,
                                            const CLBlastPrecision precision, const size_t num_parameters,
                                            const char** parameters_names, const size_t* parameters_values)
```

Arguments to OverrideParameters (C++ version):

* `const cl_device_id device`: The OpenCL device to set the new parameters for.
* `const std::string &kernel_name`: The target kernel name. This has to be one of the existing CLBlast kernels (Xaxpy, Xdot, Xgemv, XgemvFast, XgemvFastRot, Xgemv, Xger, Copy, Pad, Transpose, Padtranspose, Xgemm, or XgemmDirect). If this argument is incorrect, this function will return with the `clblast::kInvalidOverrideKernel` status-code.
* `const Precision precision`: The CLBlast precision enum to set the new parameters for.
* `const std::unordered_map<std::string,size_t> &parameters`: An unordered map of strings to integers. This has to contain all the tuning parameters for a specific kernel as reported by the included tuners (e.g. `{ {"COPY_DIMX",8}, {"COPY_DIMY",32}, {"COPY_VW",4}, {"COPY_WPT",8} }` for the `Copy` kernel). If this argument is incorrect, this function will return with the `clblast::kMissingOverrideParameter` status-code.



Tune<kernel_name>: Run the tuner for a particular kernel (advanced usage)
-------------

The CLBlast kernels can be tuned using the tuning binaries, but also programmatically through an API. This is only recommended for advanced usage, see for more information [the tuning docs](tuning.md).

C++ API:
```
// Tunes the "Xaxpy" kernel, used for many level-1 routines such as XAXPY, XCOPY, and XSWAP
template <typename T>
StatusCode PUBLIC_API TuneXaxpy(cl_command_queue* queue, const size_t n,
                                const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Xdot" kernel, used for level-1 reduction routines such as XDOT, XMAX, and XSUM
template <typename T>
StatusCode PUBLIC_API TuneXdot(cl_command_queue* queue, const size_t n,
                               const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Xgemv" kernel, used for matrix-vector level-2 routines such as XGEMV, XGBMV, and XHEMV
template <typename T>
StatusCode PUBLIC_API TuneXgemv(cl_command_queue* queue, const size_t m, const size_t n,
                                const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Xger" kernel, used for matrix update level-2 routines such as XGER, XHER, and XSYR2
template <typename T>
StatusCode PUBLIC_API TuneXger(cl_command_queue* queue, const size_t m, const size_t n,
                               const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Xgemm" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TuneXgemm(cl_command_queue* queue, const size_t m, const size_t n, const size_t k,
                               const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "XgemmDiret" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TuneXgemmDirect(cl_command_queue* queue, const size_t m, const size_t n, const size_t k,
                                      const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Copy" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TuneCopy(cl_command_queue* queue, const size_t m, const size_t n,
                               const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Pad" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TunePad(cl_command_queue* queue, const size_t m, const size_t n,
                              const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Transpose" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TuneTranspose(cl_command_queue* queue, const size_t m, const size_t n,
                                    const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Padtranspose" kernel, used for most level-3 routines such as XGEMM, XSYMM, and XHER2K
template <typename T>
StatusCode PUBLIC_API TunePadtranspose(cl_command_queue* queue, const size_t m, const size_t n,
                                       const double fraction, std::unordered_map<std::string,size_t> &parameters);

// Tunes the "Xgemm" kernel, used for the level-3 routine XTRSM
template <typename T>
StatusCode PUBLIC_API TuneInvert(cl_command_queue* queue, const size_t m, const size_t n, const size_t k,
                                 const double fraction, std::unordered_map<std::string,size_t> &parameters);
```

Arguments to Tune<kernel_name> (C++ version):

* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to tune the kernel for.
* `const size_t m`: The routine argument `m` to tune for (not applicable for all kernels)
* `const size_t n`: The routine argument `n` to tune for
* `const size_t k`: The routine argument `k` to tune for (not applicable for all kernels)
* `const double fraction`: A value between 0.0 and 1.0 which determines the fraction of the tuning search space to explore.
* `std::unordered_map<std::string,size_t> &parameters`: An unordered map of strings to integers. This will return the best found tuning parameters.
