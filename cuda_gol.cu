#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath> // For ceil
#include <cuda_runtime.h>
#include <algorithm> // For std::swap

// Macro for checking CUDA errors
#define CUDACHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

    // Read grid dimensions
    if (fscanf(file, "%zu%zu", &num_rows, &num_cols) != 2) {
        fprintf(stderr, "Error reading grid dimensions from file: %s\n", filename);
        fclose(file);
        return false;
    }

    // Calculate leading dimension (width + 2 for padding)
    ld = num_cols + 2;

    // Allocate pinned host memory for the grid (rows + 2 for padding)
    CUDACHECK(cudaHostAlloc((void**)&grid, (num_rows + 2) * ld * sizeof(int), cudaHostAllocDefault));

    // Initialize padding borders to 0 (dead)
    for(size_t c = 0; c < ld; c++) grid[c] = 0; // Top border row
    for(size_t r = 1; r <= num_rows; r++)
    {
        grid[r * ld + 0] = 0; // Left border column
        for(size_t c = 1; c <= num_cols; c++)
        {
            char in_char;
            if (fscanf(file, " %c", &in_char) != 1) {
                 fprintf(stderr, "Error reading grid data at row %zu, col %zu from file: %s\n", r, c, filename);
                 // Use cudaFreeHost for pinned memory
                 CUDACHECK(cudaFreeHost(grid));
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


__global__ void life_iteration_kernel_shared(
    const int *grid_in, 
    int *grid_out,       
    size_t num_rows,     
    size_t num_cols,     
    size_t ld)           
{
    extern __shared__ int s_tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bdx = blockDim.x; 
    int bdy = blockDim.y; 
    
    int actual_tile_width = bdx - 2;
    int actual_tile_height = bdy - 2;

    size_t block_actual_r_start = blockIdx.y * actual_tile_height;
    size_t block_actual_c_start = blockIdx.x * actual_tile_width;

    int r_glob_to_load = (int)block_actual_r_start + ty - 1; 
    int c_glob_to_load = (int)block_actual_c_start + tx - 1; 

    int s_tile_idx = ty * bdx + tx;

    if (r_glob_to_load >= 0 && r_glob_to_load < (int)num_rows &&
        c_glob_to_load >= 0 && c_glob_to_load < (int)num_cols) {
        s_tile[s_tile_idx] = grid_in[((size_t)r_glob_to_load + 1) * ld + ((size_t)c_glob_to_load + 1)];
    } else {
        s_tile[s_tile_idx] = 0; 
    }

    __syncthreads(); 

    if (tx >= 1 && tx < bdx - 1 && ty >= 1 && ty < bdy - 1) {
        int current_cell_value_in_stile = s_tile[s_tile_idx]; 
        bool is_alive = (current_cell_value_in_stile != 0);

        int live_neighbors = 0;
        live_neighbors += s_tile[(ty - 1) * bdx + (tx - 1)]; 
        live_neighbors += s_tile[(ty - 1) * bdx +  tx     ]; 
        live_neighbors += s_tile[(ty - 1) * bdx + (tx + 1)]; 
        live_neighbors += s_tile[ ty      * bdx + (tx - 1)]; 
        live_neighbors += s_tile[ ty      * bdx + (tx + 1)]; 
        live_neighbors += s_tile[(ty + 1) * bdx + (tx - 1)]; 
        live_neighbors += s_tile[(ty + 1) * bdx +  tx     ]; 
        live_neighbors += s_tile[(ty + 1) * bdx + (tx + 1)]; 

        bool will_be_alive; 
        if (is_alive) {
            will_be_alive = (live_neighbors == 2 || live_neighbors == 3);
        } else {
            will_be_alive = (live_neighbors == 3);
        }

        int inner_tile_r = ty - 1; 
        int inner_tile_c = tx - 1; 

        size_t g_r_out = block_actual_r_start + inner_tile_r;
        size_t g_c_out = block_actual_c_start + inner_tile_c;

        if (g_r_out < num_rows && g_c_out < num_cols) {
            grid_out[(g_r_out + 1) * ld + (g_c_out + 1)] = will_be_alive ? 1 : 0;
        }
    }


    

}


void simulate_life_cuda(int * grid, size_t num_rows, size_t num_cols, int num_iterations, int threads_x = 16, int threads_y = 16)
{
    size_t ld = num_cols + 2;
    size_t grid_size_bytes = (num_rows + 2) * ld * sizeof(int);

    int *d_grid_a, *d_grid_b;
    CUDACHECK(cudaMalloc(&d_grid_a, grid_size_bytes));
    CUDACHECK(cudaMalloc(&d_grid_b, grid_size_bytes));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaMemcpyAsync(d_grid_a, grid, grid_size_bytes, cudaMemcpyHostToDevice, stream));
    dim3 threads_per_block(threads_x, threads_y);
    dim3 numBlocks(
        (unsigned int)ceil((double)num_cols / (threads_per_block.x-2)),
        (unsigned int)ceil((double)num_rows / (threads_per_block.y-2)));

    
    // Need space for (threads_x + 2) * (threads_y + 2) integers for the tile + halo
    size_t shared_mem_size = (threads_per_block.x) * (threads_per_block.y) * sizeof(int);

    int *d_grid_curr = d_grid_a;
    int *d_grid_next = d_grid_b;

    CUDACHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < num_iterations; i++)
    {
        life_iteration_kernel_shared<<<numBlocks, threads_per_block, shared_mem_size, stream>>>(
            d_grid_curr, d_grid_next, num_rows, num_cols, ld
        );
        CUDACHECK(cudaGetLastError());

        std::swap(d_grid_curr, d_grid_next);
    }

    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));


    float milliseconds = 0;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("%.3f\n", milliseconds/1000.0); // ms /1000 = seconds



    CUDACHECK(cudaMemcpyAsync(grid, d_grid_curr, grid_size_bytes, cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    CUDACHECK(cudaFree(d_grid_a));
    CUDACHECK(cudaFree(d_grid_b));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamDestroy(stream));
}




int main(int argc, char ** argv)
{
    int num_iterations = 10;
    const char * input_file = "grid.txt";      // Default input file path (relative)
    const char * output_file = "grid.out.txt"; // Default output file path (relative)
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
         fprintf(stderr, "Warning: threads_x * threads_y (%d) exceeds typical limit of 1024. May fail.\n", threads_x * threads_y);
    }


    int * h_grid = nullptr; // Host grid pointer (pinned memory)
    size_t num_rows, num_cols;

    bool success_read = read_grid_from_file(input_file, &h_grid, &num_rows, &num_cols);

    if(!success_read)
    {
        fprintf(stderr, "Error: Failed to load grid from file.\n");
        CUDACHECK(cudaFreeHost(h_grid));
        return 1;
    }

    simulate_life_cuda(h_grid, num_rows, num_cols, num_iterations, threads_x, threads_y);


    bool success_write = write_grid_to_file(output_file, h_grid, num_rows, num_cols);


    if(!success_write)
    {
        fprintf(stderr, "Error: Failed to save grid to file.\n");
        CUDACHECK(cudaFreeHost(h_grid));
        return 2;
    }


 
    CUDACHECK(cudaFreeHost(h_grid));

    return 0;
}
