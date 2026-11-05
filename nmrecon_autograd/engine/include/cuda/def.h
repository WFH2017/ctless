
#define CHECK_CUDA(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)