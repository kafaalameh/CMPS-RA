#include <stdio.h>
#include <mpi.h>

#define arow 2
#define acol 2
#define brow 2
#define bcol 2
#define crow 2
#define ccol 2

// scatter matrix a to aa
// broadcast matrix b
// compute cc=aa*b
// gather cc to c

void results(const char *prompt, int a[arow][acol]);

int main(int argc, char *argv[]){
    int i, j, k, rank, size, tag = 99, blksz, sum = 0;
    int a[arow][acol]={{1, 2}, {3, 4}};
    int b[brow][bcol]={{5, 6}, {7, 8}};
    int c[crow][ccol];
    int aa[arow],cc[crow];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // scatter rows of matrix a to different processes     
    MPI_Scatter(a, arow*acol/size, MPI_INT, aa, arow*acol/size, MPI_INT, 0, MPI_COMM_WORLD);

    //broadcast matrix b to all processes
    MPI_Bcast(b, brow*bcol, MPI_INT, 0, MPI_COMM_WORLD);

    // blocks until all processes in the communicator have reached this routine
    MPI_Barrier(MPI_COMM_WORLD);

    // all processes perform vector matrix multiplication
    for(i=0; i<bcol; i++){
        for(j=0; j<brow; j++){
            sum = sum + aa[j]*b[j][i];
        }
        cc[i] = sum;
        sum = 0;
    }

    // gather rows from many processes into a single process
    MPI_Gather(cc, crow*ccol/size, MPI_INT, c, crow*ccol/size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);        
    MPI_Finalize();

    if (rank == 0) results("C = ", c);
    return 0;
}

void results(const char *prompt, int a[arow][acol]){
    int i, j;
    printf ("%s\n", prompt);
    for(i=0; i<arow; i++){
        for(j=0; j<acol; j++){
            printf("%4d", a[i][j]);
        }
        printf("\n");
    }
}