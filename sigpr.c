/* Convolution example; originally written by Lucas Wilcox.
 * Minor modifications by Georg Stadler.
 * The function expects a bitmap image (*.ppm) as input, as
 * well as a number of blurring loops to be performed.
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include "cl-helper.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void spri(float* a, float* b, int size){
  int i;
  for(i = 0; i < size; i++){
    b[i] = exp(-a[i])/((1 + exp(-a[i]))*(1 + exp(-a[i])));
  }
  return;
}

void print_kernel_info(cl_command_queue queue, cl_kernel knl)
{
  // get device associated with the queue
  cl_device_id dev;
  CALL_CL_SAFE(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(dev), &dev, NULL));

  char kernel_name[4096];
  CALL_CL_SAFE(clGetKernelInfo(knl, CL_KERNEL_FUNCTION_NAME,
        sizeof(kernel_name), &kernel_name, NULL));
  kernel_name[4095] = '\0';
  printf("Info for kernel %s:\n", kernel_name);

  size_t kernel_work_group_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(kernel_work_group_size), &kernel_work_group_size, NULL));
  printf("  CL_KERNEL_WORK_GROUP_SIZE=%zd\n", kernel_work_group_size);

  size_t preferred_work_group_size_multiple;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_work_group_size_multiple),
        &preferred_work_group_size_multiple, NULL));
  printf("  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=%zd\n",
      preferred_work_group_size_multiple);

  cl_ulong kernel_local_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL));
  printf("  CL_KERNEL_LOCAL_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_local_mem_size);

  cl_ulong kernel_private_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
        sizeof(kernel_private_mem_size), &kernel_private_mem_size, NULL));
  printf("  CL_KERNEL_PRIVATE_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_private_mem_size);
}


int main(int argc, char *argv[])
{
   // size of array
   if(argc != 2){
    printf("You need exactly two arguments!\n exit!");
   }
   int n = atoi(argv[1]);
   unsigned int size = n;
   unsigned int mem_size = sizeof(float) * size;
   float* arr = (float *) malloc(mem_size);
   float* res = (float *) malloc(mem_size);
   float* res2 = (float *) malloc(mem_size);
   int i;
   for(i = 0; i < n; i++){
      arr[i] = 1.0;
   }

  // --------------------------------------------------------------------------
  // get an OpenCL context and queue
  // --------------------------------------------------------------------------
  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);
  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("sig_pr.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "sigmoid", NULL);
  free(knl_text);

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem d_arr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_res = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      mem_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");


  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_arr, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size, arr, 0, NULL, NULL));

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_res, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size, res, 0, NULL, NULL));

  size_t local_size[] = { 1, 1 };
  size_t global_size[] = {n, 1};

  CALL_CL_SAFE(clSetKernelArg(knl, 0, sizeof(d_arr), &d_arr));
  CALL_CL_SAFE(clSetKernelArg(knl, 1, sizeof(d_res), &d_res));

  CALL_CL_SAFE(clFinish(queue));
    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 2, NULL,
          global_size, local_size, 0, NULL, NULL));

  CALL_CL_SAFE(clFinish(queue));

  CALL_CL_SAFE(clEnqueueReadBuffer(
          queue, d_res,  CL_TRUE,  0,
          mem_size, res,
          0, NULL, NULL));

    int j;
    float zsum = 0.0;

    spri(arr, res2, n);
    float temp;

    for(j = 0; j < n; j++){
      temp = res[j] - res2[j];
      if(temp < 0)
        temp = -temp;
      zsum += temp; 
    }
    printf("total difference between results from GPU and CPU is %.15f\n", zsum);


  CALL_CL_SAFE(clReleaseMemObject(d_res));
  CALL_CL_SAFE(clReleaseMemObject(d_arr));
  CALL_CL_SAFE(clReleaseKernel(knl));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));
  free(arr);
  free(res);
  free(res2);
}
//gcc -o NN n_host.c cl-helper.c helper.c -framework OpenCL -lm