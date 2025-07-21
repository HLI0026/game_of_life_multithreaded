#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <omp.h>


bool read_grid_from_file(const char * filename, int ** grid_out, size_t * num_rows_out, size_t * num_cols_out)
{
    FILE * file = fopen(filename, "r");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open input file\n");
        return false;
    }

    int * grid;
    size_t num_rows;
    size_t num_cols;
    size_t ld;

    fscanf(file, "%zu%zu", &num_rows, &num_cols);
    grid = new int[(num_rows+2) * (num_cols+2)];
    ld = num_cols + 2;

    for(size_t c = 0; c <= num_cols+1; c++) grid[c] = 0;
    for(size_t r = 1; r <= num_rows; r++)
    {
        grid[r * ld + 0] = 0;
        for(size_t c = 1; c <= num_cols; c++)
        {
            char in_char;
            fscanf(file, " %c ", &in_char);
            int grid_val = (in_char == '.') ? 0 : 1;
            grid[r * ld + c] = grid_val;
        }
        grid[r * ld + ld] = 0;
    }
    for(size_t c = 0; c <= num_cols+1; c++) grid[(num_rows+1) * ld + c] = 0;

    fclose(file);

    *grid_out = grid;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    return true;
}



void print_grid(const int * grid, size_t num_rows, size_t num_cols, bool with_borders = false, FILE * file = stdout)
{
    size_t ld = num_cols + 2;
    size_t bo = (with_borders ? 1 : 0);

    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 1 - bo; r <= num_rows + bo; r++)
    {
        for(size_t c = 1 - bo; c <= num_cols + bo; c++)
        {
            int grid_val = grid[r * ld + c];
            char out_char = (grid_val == 0) ? '.' : 'X';
            fprintf(file, "%c", out_char);
        }
        fprintf(file, "\n");
    }
}



bool write_grid_to_file(const char * filename, const int * grid, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "w");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    print_grid(grid, num_rows, num_cols, false, file);

    fclose(file);

    return true;
}



void copy_grid(const int * grid_src, int * grid_dst, size_t num_rows, size_t num_cols)
{
    size_t ld = num_cols + 2;

    for(size_t r = 0; r <= num_rows+1; r++)
    {
        for(size_t c = 0; c <= num_cols+1; c++)
        {
            grid_dst[r * ld + c] = grid_src[r * ld + c];
        }
    }
}



void life_iteration(const int * grid_in, int * grid_out, size_t num_rows, size_t num_cols)
{
    size_t ld = num_cols + 2;

    #pragma omp parallel for collapse(2) default(none) \
    shared(grid_in, grid_out, num_rows, num_cols, ld)
    for(size_t r = 1; r <= num_rows; r++)
    {
        for(size_t c = 1; c <= num_cols; c++)
        {
            bool is_alive = (grid_in[r * ld + c] != 0);
            int num_neighbours = 0;
            for(size_t row = r-1; row <= r+1; row++)
            {
                for(size_t col = c-1; col <= c+1; col++)
                {
                    num_neighbours += grid_in[row * ld + col];
                }
            }
            num_neighbours -= grid_in[r * ld + c];

            bool will_be_alive;
            if(is_alive)
            {
                if(num_neighbours < 2) will_be_alive = false;
                else if(num_neighbours > 3) will_be_alive = false;
                else will_be_alive = true;
            }
            else
            {
                if(num_neighbours == 3) will_be_alive = true;
                else will_be_alive = false;
            }
            grid_out[r * ld + c] = will_be_alive;
        }
    }
}



void simulate_life(int * grid, size_t num_rows, size_t num_cols, int num_iterations)
{
    size_t ld = num_cols + 2;

    int * grid_2 = new int[(num_rows+2) * (num_cols+2)];
    for(size_t r = 0; r < num_rows+2; r++)
    {
        for(size_t c = 0; c < num_cols+2; c++)
        {
            grid_2[r * ld + c] = 0;
        }
    }

    double time = omp_get_wtime();
    

    for(int i = 0; i < num_iterations; i++)
    {
        int * grid_curr = ((i % 2 == 0) ? grid : grid_2);
        int * grid_next = ((i % 2 == 0) ? grid_2 : grid);

        // printf("\n\n\n");
        // printf("Iteration %d\n", i);
        // print_grid(grid, num_rows, num_cols, false, stdout);
        // std::this_thread::sleep_for(std::chrono::milliseconds(200));

        life_iteration(grid_curr, grid_next, num_rows, num_cols);
    }

    printf("%f\n", omp_get_wtime() - time);

    if(num_iterations % 2 != 0)
    {
        copy_grid(grid_2, grid, num_rows, num_cols);
    }

    delete[] grid_2;
}





int main(int argc, char ** argv)
{
    // printf("Usage: ./game_of_life num_iterations input_file.txt output_file.txt\n");
    // printf("All parameters are optional and have default values\n");
    // printf("\n");

    int num_iterations = 10;
    const char * input_file = "io/grid.txt";
    const char * output_file = "io/grid.out.txt";

    if(argc > 1) num_iterations = atoi(argv[1]);
    if(argc > 2) input_file = argv[2];
    if(argc > 3) output_file = argv[3];

    // printf("Command line arguments:\n");
    // printf("  num_iterations: %d\n", num_iterations);
    // printf("  input_file:     %s\n", input_file);
    // printf("  output_file:    %s\n", output_file);
    // printf("\n");

    if(num_iterations < 0)
    {
        fprintf(stderr, "Wrong argument value\n");
        return 1;
    }



    int * grid;
    size_t num_rows;
    size_t num_cols;
    
    // printf("Loading input grid ...\n");
    bool success_read = read_grid_from_file(input_file, &grid, &num_rows, &num_cols);
    if(!success_read)
    {
        fprintf(stderr, "Failed to load grid\n");
        return 1;
    }
    // printf("Done\n");
    // printf("\n");

    // printf("Simulating life ...\n");
    simulate_life(grid, num_rows, num_cols, num_iterations);
    // printf("Done\n");
    // printf("\n");

    // printf("Writing grid to file ...\n");
    bool success_write = write_grid_to_file(output_file, grid, num_rows, num_cols);
    if(!success_write)
    {
        fprintf(stderr, "Failed to save grid\n");
        return 2;
    }
    // printf("Done\n");
    // printf("\n");

    delete[] grid;

    // printf("Finished successfully\n");

    return 0;
}
