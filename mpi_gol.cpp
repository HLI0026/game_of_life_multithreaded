#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <omp.h>
#include <mpi.h>
#include <cstring>

struct grid_info {
    int dimensions[2];
    int local_rows;
    int local_cols;
    int proc_row;
    int proc_col;
    int rank;
    int size;
    int num_rows;
    int num_cols;
    bool right_edge;
    bool bottom_edge;
    bool left_edge;
    bool top_edge;
    int iterations;
};



grid_info get_grid_info(int rank, int size, int num_rows, int num_cols,int * dimensions, int iterations)
{
    grid_info info;
    info.rank = rank;
    info.size = size;
    info.num_rows = num_rows;
    info.num_cols = num_cols;

    info.dimensions[0] = dimensions[0];
    info.dimensions[1] = dimensions[1];

    info.right_edge = (rank % info.dimensions[1] == info.dimensions[1] - 1);
    info.bottom_edge = (rank / info.dimensions[1] == info.dimensions[0] - 1);
    info.left_edge = (rank % info.dimensions[1] == 0);
    info.top_edge = (rank / info.dimensions[1] == 0);
    
    // pozice procesu v gridu
    info.proc_row = rank / info.dimensions[1];
    info.proc_col = rank % info.dimensions[1];

    // velikost submatice
    info.local_rows = num_rows / info.dimensions[0];
    info.local_cols = num_cols / info.dimensions[1];

    // pokud jsme na hranici (vpravo/dole), tak extendneme localni matici
    if (info.proc_row == info.dimensions[0] - 1) info.local_rows += num_rows % info.dimensions[0];
    if (info.proc_col == info.dimensions[1] - 1) info.local_cols += num_cols % info.dimensions[1];

    info.iterations = iterations;

    return info;
}



bool read_grid_from_file(const char * filename, int ** grid_out, int * num_rows_out, int * num_cols_out)
{
    FILE * file = fopen(filename, "r");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open input file\n");
        return false;
    }

    int * grid;
    int num_rows;
    int num_cols;
    int ld;

    fscanf(file, "%zu%zu", &num_rows, &num_cols);
    grid = new int[(num_rows+2) * (num_cols+2)];
    ld = num_cols + 2;

    for(int c = 0; c <= num_cols+1; c++) grid[c] = 0;
    for(int r = 1; r <= num_rows; r++)
    {
        grid[r * ld + 0] = 0;
        for(int c = 1; c <= num_cols; c++)
        {
            char in_char;
            fscanf(file, " %c ", &in_char);
            int grid_val = (in_char == '.') ? 0 : 1;
            grid[r * ld + c] = grid_val;
        }
        grid[r * ld + ld] = 0;
    }
    for(int c = 0; c <= num_cols+1; c++) grid[(num_rows+1) * ld + c] = 0;

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
    
    // printf("bo %d\n", bo);
    // bo = 1;
    
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



void copy_grid(const int * grid_src, int * grid_dst, grid_info info)
{
    int ld = info.local_cols + 2;

    for(int r = 0; r <= info.local_rows+1; r++)
    {
        for(int c = 0; c <= info.local_cols+1; c++)
        {
            grid_dst[r * ld + c] = grid_src[r * ld + c];
        }
    }
    // for(int r = 0; r < info.local_rows+1; r++)
    // {
        // for(int c = 0; c < info.local_cols+1; c++)
        // {
            // grid_dst[r * (info.num_cols+2) + c] = grid_src[r * (info.local_cols+2) + c];
        // }
    // }

}



void life_iteration(const int * grid_in, int * grid_out, grid_info info)
{
    int ld = info.local_cols + 2;
    #pragma omp parallel for collapse(2) default(none) \
    shared(grid_in, grid_out, info, ld)
    for(int r = 1; r <= info.local_rows; r++)
    {
        for(int c = 1; c <= info.local_cols; c++)
        {
            bool is_alive = (grid_in[r * ld + c] != 0);
            int num_neighbours = 0;
            for(int row = r-1; row <= r+1; row++)
            {
                for(int col = c-1; col <= c+1; col++)
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



void simulate_life(int * local_grid, grid_info info)
{
    
    int l_sizes_send[2] = {info.local_rows+2, info.local_cols+2};
    int l_subsizes_send[2] = {info.local_rows, 1};
    int l_starts_send[2] = {1, 1};
    
    int r_sizes_send[2] = {info.local_rows+2, info.local_cols+2};
    int r_subsizes_send[2] = {info.local_rows, 1};
    int r_starts_send[2] = {1, info.local_cols+2-2};
    
    int t_sizes_send[2] = {info.local_rows+2, info.local_cols+2};
    int t_subsizes_send[2] = {1, info.local_cols};
    int t_starts_send[2] = {1, 1};
    
    int b_sizes_send[2] = {info.local_rows+2, info.local_cols+2};
    int b_subsizes_send[2] = {1, info.local_cols};
    int b_starts_send[2] = {info.local_rows+2-2, 1};
    
    MPI_Datatype left_row_send;
    MPI_Datatype right_row_send;
    MPI_Datatype top_row_send;
    MPI_Datatype bottom_row_send;

    MPI_Type_create_subarray(2, l_sizes_send, l_subsizes_send, l_starts_send, MPI_ORDER_C, MPI_INT, &left_row_send);
    MPI_Type_create_subarray(2, r_sizes_send, r_subsizes_send, r_starts_send, MPI_ORDER_C, MPI_INT, &right_row_send);
    MPI_Type_create_subarray(2, t_sizes_send, t_subsizes_send, t_starts_send, MPI_ORDER_C, MPI_INT, &top_row_send);
    MPI_Type_create_subarray(2, b_sizes_send, b_subsizes_send, b_starts_send, MPI_ORDER_C, MPI_INT, &bottom_row_send);
    
    MPI_Type_commit(&left_row_send);
    MPI_Type_commit(&right_row_send);
    MPI_Type_commit(&top_row_send);
    MPI_Type_commit(&bottom_row_send);

    int l_sizes_recv[2] = {info.local_rows+2, info.local_cols+2};
    int l_subsizes_recv[2] = {info.local_rows, 1};
    int l_starts_recv[2] = {1, 0};
    int r_sizes_recv[2] = {info.local_rows+2, info.local_cols+2};
    int r_subsizes_recv[2] = {info.local_rows, 1};
    int r_starts_recv[2] = {1, info.local_cols+2-1};
    int t_sizes_recv[2] = {info.local_rows+2, info.local_cols+2};
    int t_subsizes_recv[2] = {1, info.local_cols};
    int t_starts_recv[2] = {0, 1};
    int b_sizes_recv[2] = {info.local_rows+2, info.local_cols+2};
    int b_subsizes_recv[2] = {1, info.local_cols};
    int b_starts_recv[2] = {info.local_rows+2-1, 1};
    
    MPI_Datatype left_row_recv;
    MPI_Datatype right_row_recv;
    MPI_Datatype top_row_recv;
    MPI_Datatype bottom_row_recv;

    MPI_Type_create_subarray(2, l_sizes_recv, l_subsizes_recv, l_starts_recv, MPI_ORDER_C, MPI_INT, &left_row_recv);
    MPI_Type_create_subarray(2, r_sizes_recv, r_subsizes_recv, r_starts_recv, MPI_ORDER_C, MPI_INT, &right_row_recv);
    MPI_Type_create_subarray(2, t_sizes_recv, t_subsizes_recv, t_starts_recv, MPI_ORDER_C, MPI_INT, &top_row_recv);
    MPI_Type_create_subarray(2, b_sizes_recv, b_subsizes_recv, b_starts_recv, MPI_ORDER_C, MPI_INT, &bottom_row_recv);
    
    MPI_Type_commit(&left_row_recv);
    MPI_Type_commit(&right_row_recv);
    MPI_Type_commit(&top_row_recv);
    MPI_Type_commit(&bottom_row_recv);

    
    
    int ld = info.local_cols + 2;

    int * grid_2 = new int[(info.local_rows+2) * ( info.local_cols+2)];
    for(int r = 0; r < info.local_rows+2; r++)
    {
        for(int c = 0; c < info.local_cols+2; c++)
        {
            grid_2[r * ld + c] = 0;
        }
    }
    // int rank_chck= 0;
    // if (info.rank == rank_chck){
        // for (int r = 0; r < info.local_rows+2; r++)
        // {
            // for(int c = 0; c < info.local_cols+2; c++)
            // {
                // printf("%d ", local_grid[r * (info.local_cols+2) + c]);
            // }
            // printf("\n");
        // }
    // printf("\n");
    // }
    // printf("dims: %d %d\n", info.dimensions[0], info.dimensions[1]);
    MPI_Barrier(MPI_COMM_WORLD);
    double beg, end;
    beg = MPI_Wtime();
    for(int i = 0; i < info.iterations; i++)
    {
        
        // printf("rank %d edges: r %d l %d t %d b %d\n",info.rank, info.right_edge, info.left_edge, info.top_edge, info.bottom_edge);
        int * grid_curr = ((i % 2 == 0) ? local_grid : grid_2);
        int * grid_next = ((i % 2 == 0) ? grid_2 : local_grid);

        // printf("\n\n\n");
        // print_grid(grid, num_rows, num_cols, false, stdout);
        // std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        // Budeme checkovat kde (ne)jsme v gridu, abychom mohli pote poslat na spravny rank/stranu
        int requests_count = !info.right_edge + !info.left_edge + !info.top_edge + !info.bottom_edge +
                             (!info.top_edge && !info.left_edge) + (!info.top_edge && !info.right_edge) +
                             (!info.bottom_edge && !info.left_edge) + (!info.bottom_edge && !info.right_edge);
        MPI_Request recv_requests[requests_count];
        MPI_Request send_requests[requests_count];
        int counter = 0;
        if (!info.right_edge){
            // printf("from rank %d to rank %d\n", info.rank, info.rank + 1);
            MPI_Isend(&grid_curr[0], 1, right_row_send, info.rank + 1, 0, MPI_COMM_WORLD, send_requests + counter);
            MPI_Irecv(&grid_curr[0], 1, right_row_recv, info.rank + 1 , 0, MPI_COMM_WORLD, recv_requests + counter);
            counter++;
// 
        }
        if (!info.left_edge){
            // printf("from rank %d to rank %d\n", info.rank, info.rank - 1);
            MPI_Isend(&grid_curr[0], 1, left_row_send, info.rank - 1, 0, MPI_COMM_WORLD, send_requests + counter);
            MPI_Irecv(&grid_curr[0], 1, left_row_recv, info.rank - 1 , 0, MPI_COMM_WORLD, recv_requests + counter);
            counter++;
        }
        if (!info.top_edge){
            // printf("topfrom rank %d to rank %d\n", info.rank, info.rank - info.dimensions[1]);
            MPI_Isend(&grid_curr[0], 1, top_row_send, info.rank - info.dimensions[1], 0, MPI_COMM_WORLD, send_requests + counter);
            MPI_Irecv(&grid_curr[0], 1, top_row_recv, info.rank - info.dimensions[1], 0, MPI_COMM_WORLD,  recv_requests + counter);
            counter++;
        }
        if (!info.bottom_edge){
            // printf("bottom from rank %d to rank %d\n", info.rank, info.rank + info.dimensions[1]);
            MPI_Isend(&grid_curr[0], 1, bottom_row_send, info.rank + info.dimensions[1], 0, MPI_COMM_WORLD,send_requests + counter);
            MPI_Irecv(&grid_curr[0], 1, bottom_row_recv, info.rank + info.dimensions[1], 0, MPI_COMM_WORLD,  recv_requests + counter);
            counter++;
        }
        if (!info.top_edge && !info.left_edge){
            // printf("from rank %d to rank %d\n", info.rank, info.rank - info.dimensions[1] - 1);
            MPI_Isend(&grid_curr[ld+1], 1, MPI_INT, info.rank - info.dimensions[1] - 1, 0, MPI_COMM_WORLD, send_requests + counter);
            MPI_Irecv(&grid_curr[0], 1, MPI_INT, info.rank - info.dimensions[1] - 1, 0, MPI_COMM_WORLD,  recv_requests + counter);
            counter++;
        }
        if (!info.top_edge && !info.right_edge){
            // printf("%d",grid_curr[info.local_cols+2-1]);
            MPI_Isend(&grid_curr[2*ld - 2], 1, MPI_INT, info.rank - info.dimensions[1] + 1, 0, MPI_COMM_WORLD,send_requests + counter);
            MPI_Irecv(&grid_curr[ld-1], 1, MPI_INT, info.rank - info.dimensions[1] + 1, 0, MPI_COMM_WORLD,  recv_requests + counter);
            // printf("%d",grid_curr[info.local_cols+2-1]);
            counter++;
        }
        if (!info.bottom_edge && !info.left_edge){
            // printf("%d",grid_curr[0]);
            MPI_Isend(&grid_curr[ld*(info.local_rows)+1], 1, MPI_INT, info.rank + info.dimensions[1] - 1, 0, MPI_COMM_WORLD,  send_requests + counter);
            MPI_Irecv(&grid_curr[ld*(info.local_rows+1)], 1, MPI_INT, info.rank + info.dimensions[1] - 1, 0, MPI_COMM_WORLD, recv_requests + counter);
            counter++;
        }
        if (!info.bottom_edge && !info.right_edge){
            // printf("%d",grid_curr[info.local_cols+2-1]);
            MPI_Isend(&grid_curr[ld*(info.local_rows+1)-2], 1, MPI_INT, info.rank + info.dimensions[1] + 1, 0, MPI_COMM_WORLD,  send_requests + counter);
            MPI_Irecv(&grid_curr[ld*(info.local_rows+2)-1], 1, MPI_INT, info.rank + info.dimensions[1] + 1, 0, MPI_COMM_WORLD,  recv_requests + counter);
            counter++;
        }
        MPI_Waitall(requests_count, send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(requests_count, recv_requests, MPI_STATUSES_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // printf("\n");
        // 
        // if (info.rank == rank_chck){
            // for (int r = 0; r < info.local_rows+2; r++)
            // {
                // for(int c = 0; c < info.local_cols+2; c++)
                // {
                    // printf("%d ", grid_curr[r * (info.local_cols+2) + c]);
                // }
                // printf("\n");
            // }
        // printf("\n");
        // }

        life_iteration(grid_curr, grid_next, info);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (info.rank==0){
        printf("%f\n", end-beg);
    }
    // if (info.rank == 0){
        // for (int r = 0; r < info.local_rows+2; r++)
        // {
            // for(int c = 0; c < info.local_cols+2; c++)
            // {
                // printf("%d ", grid_2[r * (info.local_cols+2) + c]);
            // }
            // printf("\n");
        // }
    // }

    if(info.iterations % 2 != 0)
    {
        copy_grid(grid_2, local_grid, info);
    }
    MPI_Type_free(&left_row_send);
    MPI_Type_free(&right_row_send);
    MPI_Type_free(&top_row_send);
    MPI_Type_free(&bottom_row_send);
    MPI_Type_free(&left_row_recv);
    MPI_Type_free(&right_row_recv);
    MPI_Type_free(&top_row_recv);
    MPI_Type_free(&bottom_row_recv);
    
    // if (info.rank == 0){
    // for (int r = 0; r < info.local_rows+2; r++)
    // {
    //     for(int c = 0; c < info.local_cols+2; c++)
    //     {
    //         printf("%d ", local_grid[r * (info.local_cols+2) + c]);
    //     }
    //     printf("\n");
    // }
    // }

    delete[] grid_2;
}





int main(int argc, char ** argv)
{

    int required = MPI_THREAD_FUNNELED, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (required > provided)
    {
        MPI_Finalize();
        fprintf(stderr, "Unable to run multitred aplication\n");
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_iterations = 10;
    const char * input_file = "io/grid.txt";
    const char * output_file = "io/grid.out.txt";

    if(argc > 1) num_iterations = atoi(argv[1]);
    if(argc > 2) input_file = argv[2];
    if(argc > 3) output_file = argv[3];


    if(num_iterations < 0)
    {
        fprintf(stderr, "Wrong argument value\n");
        return 1;
    }

    // printf("Command line arguments:\n");
    // printf("  num_iterations: %d\n", num_iterations);
    // printf("  input_file:     %s\n", input_file);
    // printf("  output_file:    %s\n", output_file);
    // printf("\n");


    int * grid;
    int num_rows;
    int num_cols;
    if (rank == 0)
    {
        bool success_read = read_grid_from_file(input_file, &grid, &num_rows, &num_cols);
        if(!success_read)
        {
            fprintf(stderr, "Failed to load grid\n");
            return 1;
        }
    }
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    grid_info info = get_grid_info(rank, size, num_rows, num_cols, dims, num_iterations);


    MPI_Datatype mtx;
    int sizes[2] = {info.local_rows+2, info.local_cols+2};
    int subsizes[2] = {info.local_rows, info.local_cols};
    int starts[2] = {1, 1};
    if (info.local_rows == 0 || info.local_cols == 0)
    {
        fprintf(stderr, "Invalid local grid size, fix by lowering number of processes or give larger grid!\n");
        return 1;
    }
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mtx);
    MPI_Type_commit(&mtx);
    
    int * local_grid = new int[(info.local_rows+2) * (info.local_cols+2)];
    //just set the value to 0, better for debugging
    memset(local_grid, 0, (info.local_rows + 2) * (info.local_cols + 2) * sizeof(int));
    
    if(rank == 0){
        MPI_Request* send_requests = nullptr;
        if (size > 1) {
            send_requests = new MPI_Request[size - 1];
        }
        for(int r = 0; r < info.local_rows+1; r++)
        {
            for(int c = 0; c < info.local_cols+1; c++)
            {
                local_grid[r* (info.local_cols+2) + c] = grid[r * (info.num_cols+2) + c];
                // printf("%d ", local_grid[r * (info.local_cols+2) + c]);
            }
        }
    
        for(int i = 1; i < size; i++){
            grid_info recv_info = get_grid_info(i, size, num_rows, num_cols, dims, num_iterations);
            int starts[2];
            starts[0] = recv_info.proc_row * (num_rows / dims[0])+1;  // row offset
            starts[1] = recv_info.proc_col * (num_cols / dims[1])+1;  // col offset
            int subsizes[2]    = {recv_info.local_rows, recv_info.local_cols};
            int  full_dims[2]  = {num_rows+2, num_cols+2};
            MPI_Datatype dtype;
            MPI_Type_create_subarray(2, full_dims, subsizes, starts, MPI_ORDER_C, MPI_INT, &dtype);
            MPI_Type_commit(&dtype);
            // vezmeme pozici zacatku gridu a posleme submatici
            // printf("sending to %d \n",i);

            MPI_Isend(&grid[0], 1, dtype, i, 0, MPI_COMM_WORLD, &send_requests[i-1]);
            MPI_Type_free(&dtype);
        }
        if (size > 1) {
            MPI_Waitall(size - 1, send_requests, MPI_STATUSES_IGNORE);
            delete[] send_requests;  // Free memory
        }
    }else // mimo rank 0 jen recievneme
        {
            // printf("recv %d\n",rank);
            MPI_Request recv_request;
            MPI_Irecv(local_grid, 1, mtx, 0, 0, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            
    }
    // just set the value to 0, better for debugging 
    //(or just finding out if it even outputs same values with 0 iters)
    
    if (rank==0) memset(grid, 0, (num_rows+2) * (num_cols+2) * sizeof(int));

    // for (int i = 0; i < info.size; i++)
    // {
    // char OF[50];  // Ensure this buffer is large enough
    // sprintf(OF, "io/grid_r%d.out.txt", rank);
    // bool success_write = write_grid_to_file(OF, local_grid, info.local_rows, info.local_cols);
    // }
    simulate_life(local_grid, info);

    
    if(rank == 0){
        MPI_Request* recv_request = nullptr;
        if (size > 1) {
            recv_request = new MPI_Request[size - 1];
        }
        
        for(int r = 0; r < info.local_rows+1; r++)
        {
            for(int c = 0; c < info.local_cols+1; c++)
            {
                grid[r * (info.num_cols+2) + c] = local_grid[r * (info.local_cols+2) + c];
            }
        }
                

        for(int i = 1; i < size; i++){
            grid_info recv_info = get_grid_info(i, size, num_rows, num_cols, dims, num_iterations);
            int starts[2];
            starts[0] = recv_info.proc_row * (num_rows / dims[0])+1;  // row offset
            starts[1] = recv_info.proc_col * (num_cols / dims[1])+1;  // col offset
            int subsizes[2]    = {recv_info.local_rows, recv_info.local_cols};
            int  full_dims[2]  = {num_rows+2, num_cols+2};
            MPI_Datatype dtype;
            
            
            MPI_Type_create_subarray(2, full_dims, subsizes, starts, MPI_ORDER_C, MPI_INT, &dtype);
            MPI_Type_commit(&dtype);
            // vezmeme pozici zacatku gridu a posleme submatici

            MPI_Irecv(&grid[0], 1, dtype, i, 0, MPI_COMM_WORLD, &recv_request[i-1]);
            MPI_Type_free(&dtype);
        }
        if (size > 1) {
            MPI_Waitall(size - 1, recv_request, MPI_STATUSES_IGNORE);
            delete[] recv_request;  // Free memory
        }
    }else // mimo rank 0 jen recievneme
        {
            MPI_Request send_reqest;
            MPI_Isend(local_grid, 1, mtx, 0, 0, MPI_COMM_WORLD, &send_reqest);
            MPI_Wait(&send_reqest, MPI_STATUS_IGNORE);
            
    }

    // for (int r = 0; r < info.local_rows+1; r++)
    // {
        // for(int c = 0; c < info.local_cols+1; c++)
        // {
            // printf("%d ", local_grid[r * (info.local_cols+2) + c]);
        // }
        // printf("\n");
    // }

    delete[] local_grid;
    if (rank == 0)
    {
        bool success_write = write_grid_to_file(output_file, grid, num_rows, num_cols);
        if(!success_write)
        {
            fprintf(stderr, "Failed to save grid\n");
            return 2;
        }

    }

    if (rank == 0) delete[] grid;
    MPI_Type_free(&mtx);
    // printf("Finished successfully\n");
    MPI_Finalize();
    return 0;
}

