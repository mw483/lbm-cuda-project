#ifndef MACROCUDA_H_
#define MACROCUDA_H_

//#include <cutil.h>


// user define
#ifdef CUDA_SAFE_CALL
#undef CUDA_SAFE_CALL
#endif

#ifdef CUDA_SAFE_CALL_NO_SYNC
#undef CUDA_SAFE_CALL_NO_SYNC
#endif


#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
        cudaError err = call;                                                    \
        if( cudaSuccess != err) {                                                \
                    fprintf(stderr, "Cuda error in file '%s' in line %i : %s. hostname = %s\n",        \
                                            __FILE__, __LINE__, cudaGetErrorString( err) ,getenv("HOSTNAME"));              \
                    exit(EXIT_FAILURE);                                                  \
                } } while (0)


#  define CUDA_SAFE_CALL( call) do {                                        \
        CUDA_SAFE_CALL_NO_SYNC(call);                                            \
        cudaError err = cudaThreadSynchronize();                                 \
        if( cudaSuccess != err) {                                                \
                    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.hostname = %s\n",        \
                                            __FILE__, __LINE__, cudaGetErrorString( err) ,getenv("HOSTNAME"));              \
                    exit(EXIT_FAILURE);                                                  \
                } } while (0)


# define CHECK_CUDA_ERROR(errorMessage) {                                \
    cudaError_t err = cudaGetLastError();                           \
    if( cudaSuccess != err) {                                       \
        fprintf(stderr, "CUDA error [%s]: %s in file '%s' in line %i : %s.\n", \
                getenv("HOSTNAME"), errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}


#endif
