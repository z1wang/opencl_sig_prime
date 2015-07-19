/* kernel.cl 
 * B = sigprime(A)
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
sigmoid(__global float* a, 
          __global float* b)
{
  int i = get_global_id(0);
  b[i] = exp(-a[i])/((1 + exp(-a[i]))*(1 + exp(-a[i])));
}