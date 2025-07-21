#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath> 
#include <hip/hip_runtime.h> 
#include <algorithm> 

#define HIPCHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error in %s at line %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


bool read_grid_from_file(const char * filename, int ** grid_out, size_t * num_rows_out, size_t * num_cols_out)
{
    FILE * file = fopen(filename, "r");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open input file: %s\n", filename);
        return false;
    }

    int * grid;
    size_t num_rows;
    size_t num_cols;
    size_t ld; // Leading dimension (pitch) including padding

    if (fscanf(file, "%zu%zu", &num_rows, &num_cols) != 2) {
        fprintf(stderr, "Error reading grid dimensions from file: %s\n", filename);
        fclose(file);
        return false;
    }

    ld = num_cols + 2;

    HIPCHECK(hipHostMalloc((void**)&grid, (num_rows + 2) * ld * sizeof(int), hipHostMallocDefault));

    for(size_t c = 0; c < ld; c++) grid[c] = 0; // Top border row
    for(size_t r = 1; r <= num_rows; r++)
    {
        grid[r * ld + 0] = 0; // Left border column
        for(size_t c = 1; c <= num_cols; c++)
        {
            char in_char;
            // Read cell state, skipping whitespace
            if (fscanf(file, " %c", &in_char) != 1) {
                 fprintf(stderr, "Error reading grid data at row %zu, col %zu from file: %s\n", r, c, filename);
                 // Use hipHostFree for pinned memory
                 HIPCHECK(hipHostFree(grid));
                 fclose(file);
                 return false;
            }
            // Convert character to 0 (dead) or 1 (alive)
            grid[r * ld + c] = (in_char == '.') ? 0 : 1;
        }
        grid[r * ld + (ld - 1)] = 0; // Right border column (index is ld-1)
    }
    for(size_t c = 0; c < ld; c++) grid[(num_rows + 1) * ld + c] = 0; // Bottom border row


    fclose(file);

    // Return allocated grid and dimensions
    *grid_out = grid;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    return true;
}


void print_grid(const int * grid, size_t num_rows, size_t num_cols, bool with_borders = false, FILE * file = stdout)
{
    size_t ld = num_cols + 2; // Leading dimension including padding
    // Determine start and end indices based on whether borders are included
    size_t r_start = with_borders ? 0 : 1;
    size_t r_end = with_borders ? num_rows + 1 : num_rows;
    size_t c_start = with_borders ? 0 : 1;
    size_t c_end = with_borders ? num_cols + 1 : num_cols;

    // Print dimensions if not printing borders (matches input format)
    if (!with_borders) {
        fprintf(file, "%zu %zu\n", num_rows, num_cols);
    }

    // Iterate through the specified grid range
    for(size_t r = r_start; r <= r_end; r++)
    {
        for(size_t c = c_start; c <= c_end; c++)
        {
            // Get cell value and convert to character representation
            int grid_val = grid[r * ld + c];
            char out_char = (grid_val == 0) ? '.' : 'X';
            fprintf(file, "%c", out_char);
        }
        fprintf(file, "\n"); // Newline after each row
    }
}


bool write_grid_to_file(const char * filename, const int * grid, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "w");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return false;
    }

    print_grid(grid, num_rows, num_cols, false, file);

    fclose(file);

    return true;
}

__global__ void life_iteration_kernel_shared(const int * grid_in, int * grid_out, size_t num_rows, size_t num_cols, size_t ld)
{
    // hipSharedMem is an alias for __shared__ in HIP, but __shared__ is standard
    extern __shared__ int s_tile[];

    size_t c = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_width = blockDim.x + 2; // blockDim.x + halo_left + halo_right

    int s_idx_center = (ty + 1) * tile_width + (tx + 1);

    if (r <= num_rows && c <= num_cols) {
        s_tile[s_idx_center] = grid_in[r * ld + c];
    } else {
        s_tile[s_idx_center] = 0; // Out of bounds, set to 0 (padding)
    }


    // Left
    if (tx == 0) {
        // Check if the corresponding global column (c-1) is within the logical grid bounds (>=1 excludes global padding)
        // Also check if the global row 'r' is within logical bounds
        if (r <= num_rows && (c - 1) >= 1) {
             // Load from global memory: grid_in[r * ld + (c - 1)]
             // Store into shared memory at index: (ty + 1) * tile_width + 0
             s_tile[(ty + 1) * tile_width + 0] = grid_in[r * ld + (c - 1)];
        } else {
             // If outside logical bounds, load 0 (padding)
             s_tile[(ty + 1) * tile_width + 0] = 0;
        }
    }
    // Right
    if (tx == blockDim.x - 1) {
        if (r <= num_rows && (c + 1) <= num_cols) {
            s_tile[(ty + 1) * tile_width + (blockDim.x + 1)] = grid_in[r * ld + (c + 1)];
        } else {
            s_tile[(ty + 1) * tile_width + (blockDim.x + 1)] = 0;
        }
    }
    // Top
    if (ty == 0) {
        if ((r - 1) >= 1 && c <= num_cols) {
            s_tile[0 * tile_width + (tx + 1)] = grid_in[(r - 1) * ld + c];
        } else {
             s_tile[0 * tile_width + (tx + 1)] = 0;
        }
    }
    // Bottom
    if (ty == blockDim.y - 1) {
        if ((r + 1) <= num_rows && c <= num_cols) {
            s_tile[(blockDim.y + 1) * tile_width + (tx + 1)] = grid_in[(r + 1) * ld + c];
        } else {
            s_tile[(blockDim.y + 1) * tile_width + (tx + 1)] = 0;
        }
    }
    // Top-left corner 
    if (tx == 0 && ty == 0) {
        if ((r - 1) >= 1 && (c - 1) >= 1) {
             s_tile[0 * tile_width + 0] = grid_in[(r - 1) * ld + (c - 1)];
        } else {
             s_tile[0 * tile_width + 0] = 0;
        }
    }
    // Top-right corner
    if (tx == blockDim.x - 1 && ty == 0) {
         if ((r - 1) >= 1 && (c + 1) <= num_cols) {
            s_tile[0 * tile_width + (blockDim.x + 1)] = grid_in[(r - 1) * ld + (c + 1)];
         } else {
            s_tile[0 * tile_width + (blockDim.x + 1)] = 0;
         }
    }
    // Bottom-left corner 
    if (tx == 0 && ty == blockDim.y - 1) {
        if ((r + 1) <= num_rows && (c - 1) >= 1) {
            s_tile[(blockDim.y + 1) * tile_width + 0] = grid_in[(r + 1) * ld + (c - 1)];
        } else {
             s_tile[(blockDim.y + 1) * tile_width + 0] = 0;
        }
    }
    // Bottom-right corner 
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
         if ((r + 1) <= num_rows && (c + 1) <= num_cols) {
            s_tile[(blockDim.y + 1) * tile_width + (blockDim.x + 1)] = grid_in[(r + 1) * ld + (c + 1)];
         } else {
             s_tile[(blockDim.y + 1) * tile_width + (blockDim.x + 1)] = 0;
         }
    }

    // Synchronize all threads in block
    __syncthreads();

    if (r > num_rows || c > num_cols) {
        return;
    }
    int current_state = s_tile[s_idx_center];
    bool is_alive = (current_state != 0);

    int local_r = ty + 1;
    int local_c = tx + 1;

    // Count live neighbors using data from the shared memory tile.
    int num_neighbours = 0;
    num_neighbours += s_tile[(local_r - 1) * tile_width + (local_c - 1)]; // Top-left
    num_neighbours += s_tile[(local_r - 1) * tile_width + local_c      ]; // Top-center
    num_neighbours += s_tile[(local_r - 1) * tile_width + (local_c + 1)]; // Top-right
    num_neighbours += s_tile[ local_r  * tile_width + (local_c - 1)]; // Middle-left
    num_neighbours += s_tile[ local_r  * tile_width + (local_c + 1)]; // Middle-right
    num_neighbours += s_tile[(local_r + 1) * tile_width + (local_c - 1)]; // Bottom-left
    num_neighbours += s_tile[(local_r + 1) * tile_width + local_c      ]; // Bottom-center
    num_neighbours += s_tile[(local_r + 1) * tile_width + (local_c + 1)]; // Bottom-right

    bool will_be_alive;
    if (is_alive) {
        if (num_neighbours < 2 || num_neighbours > 3) {
            will_be_alive = false;
        }
        else {
            will_be_alive = true;
        }
    } else {
        if (num_neighbours == 3) {
            will_be_alive = true;
        } else {
            // Stays dead otherwise
            will_be_alive = false;
        }
    }

    grid_out[r * ld + c] = will_be_alive ? 1 : 0;

}



void simulate_life_hip(int * grid, size_t num_rows, size_t num_cols, int num_iterations, int threads_x = 16, int threads_y = 16)
{
    size_t ld = num_cols + 2;
    size_t grid_size_bytes = (num_rows + 2) * ld * sizeof(int);

    int *d_grid_a, *d_grid_b;
    HIPCHECK(hipMalloc(&d_grid_a, grid_size_bytes));
    HIPCHECK(hipMalloc(&d_grid_b, grid_size_bytes));

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    hipEvent_t start, stop;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));

    HIPCHECK(hipMemcpyAsync(d_grid_a, grid, grid_size_bytes, hipMemcpyHostToDevice, stream));

    dim3 threads_per_block(threads_x, threads_y);
    dim3 numBlocks(
        (unsigned int)ceil((double)num_cols / threads_per_block.x),
        (unsigned int)ceil((double)num_rows / threads_per_block.y)
    );


    size_t shared_mem_size = (threads_per_block.x + 2) * (threads_per_block.y + 2) * sizeof(int);
    int *d_grid_curr = d_grid_a;
    int *d_grid_next = d_grid_b;

    HIPCHECK(hipEventRecord(start, stream));

    for (int i = 0; i < num_iterations; i++)
    {
        life_iteration_kernel_shared<<<numBlocks, threads_per_block, shared_mem_size, stream>>>(
            d_grid_curr, d_grid_next, num_rows, num_cols, ld
        );
        HIPCHECK(hipGetLastError());

        std::swap(d_grid_curr, d_grid_next);
    }

    HIPCHECK(hipEventRecord(stop, stream));
    HIPCHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIPCHECK(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Simulation Time: %.3f seconds\n", milliseconds / 1000.0);


    HIPCHECK(hipMemcpyAsync(grid, d_grid_curr, grid_size_bytes, hipMemcpyDeviceToHost, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    HIPCHECK(hipFree(d_grid_a));
    HIPCHECK(hipFree(d_grid_b));
    HIPCHECK(hipEventDestroy(start));
    HIPCHECK(hipEventDestroy(stop));
    HIPCHECK(hipStreamDestroy(stream));
}




int main(int argc, char ** argv)
{
    int num_iterations = 10;
    const char * input_file = "grid.txt";       // Default input file path
    const char * output_file = "grid.out.txt";  // Default output file path
    int threads_x = 16; // Default block dimension x
    int threads_y = 16; // Default block dimension y

    if(argc > 1) num_iterations = atoi(argv[1]);
    if(argc > 2) input_file = argv[2];
    if(argc > 3) output_file = argv[3];
    if(argc > 4) threads_x = atoi(argv[4]);
    if(argc > 5) threads_y = atoi(argv[5]);

    if(num_iterations < 0)
    {
        fprintf(stderr, "Error: Number of iterations must be non-negative.\n");
        return 1;
    }
    if(threads_x <= 0 || threads_y <= 0) {
        fprintf(stderr, "Error: Threads per block must be positive.\n");
        return 1;
    }
    if (threads_x * threads_y > 1024) {
         fprintf(stderr, "Warning: threads_x * threads_y (%d) exceeds typical limit of 1024. May cause issues.\n", threads_x * threads_y);
    }


    int * h_grid = nullptr;
    size_t num_rows, num_cols; // Grid dimensions (logical, excluding padding)

    bool success_read = read_grid_from_file(input_file, &h_grid, &num_rows, &num_cols);

    if(!success_read)
    {
        fprintf(stderr, "Error: Failed to load grid from file.\n");
        HIPCHECK(hipHostFree(h_grid));
        return 1;
    }

    simulate_life_hip(h_grid, num_rows, num_cols, num_iterations, threads_x, threads_y);


    bool success_write = write_grid_to_file(output_file, h_grid, num_rows, num_cols);


    if(!success_write)
    {
        fprintf(stderr, "Error: Failed to save grid to file.\n");
        HIPCHECK(hipHostFree(h_grid));
        return 2;
    }



    HIPCHECK(hipHostFree(h_grid));

    return 0; // Success
}
