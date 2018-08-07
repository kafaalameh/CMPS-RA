#include <stdio.h> /* C library to perform Input/Output operations */
#include <stdlib.h> /* C Standard General Utilities Library*/
#include <math.h> /* Declares a set of functions to compute common mathematical operations and transformations */
#include <mpi.h> /* MPI Library */
#include <time.h> /* contains definitions of functions to get and manipulate date and time information */

#define N 2 /* define the dimension of the input matrix */

int matrixA[N][N]; // input matrix A
int matrixB[N][N]; // input matrix B

/*
	We need to identify the processes in MPI_COMM_WORLD with the 
	coordinates of the square grid, and each row and each column
	of the grid needs to form its own communicator.
	Hence, we need to associate a square grid structure with MPI_COMM_WORLD
	*/

typedef struct {
	int p; /* number of processors */
	MPI_Comm comm; /* handle to global grid communicator */
	MPI_Comm row_comm; /* row communicator */
	MPI_Comm col_comm; /* column communicator */
	int q; /* dimension of the grid, = sqrt(p) */
	int my_row; /* row position of a processor in a grid */
	int my_col; /* column position of a procesor in a grid */
	int my_rank; /* rank within the grid */
}GridInfo;


void SetupGrid(GridInfo *grid) {
	int old_rank; /* old rank of processor */
	int dimensions[2]; /* number of dimension in the grid, we have two*/
	int wrap_around[2]; /* periodicity: if the first entry if this row or column
	is adjacent to the last entry in that row or column. In fox's alg, we require
	the 2nd dimension to be periodic since we want a circular shift of the submatrices
	in each column*/
	int coordinates[2]; /* coordinate of the grid */
	int free_coords[2]; /* coordinate of the partitioned grid*/
	
	/* get the overall information before overlaying cart_grid */

	/*
	Determines the size of the group associated with a communicator
	int MPI_Comm_size( MPI_Comm comm, int *size ) 
	*/
	MPI_Comm_size(MPI_COMM_WORLD,&(grid->p));

	/*
	Determines the rank of the calling process in the communicator
	int MPI_Comm_rank( MPI_Comm comm, int *rank ) 
	*/
	MPI_Comm_rank(MPI_COMM_WORLD,&old_rank);
	
	/* Assumption: p is a perfect square */
	grid->q=(int)sqrt((double)grid->p);
	/* set the dimensions */
	dimensions[0]=dimensions[1]=grid->q;
	
	/* we want a torus on the second dimension, so set it appropriately */

	wrap_around[0]=0;
	wrap_around[1]=1;
	
	/*
	Makes a new communicator to which topology information has been attached
	int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart)
	*/
	MPI_Cart_create(MPI_COMM_WORLD,2,dimensions,wrap_around,1,&(grid->comm));
	/* since we have set reorder to true, this might have changed the ranks */
	
	/*
	Determines the rank of the calling process in the communicator
	int MPI_Comm_rank( MPI_Comm comm, int *rank ) 
	*/
	MPI_Comm_rank(grid->comm,&(grid->my_rank));
	/* get the cartesian coordinates for the current process */
	
	/*
	Determines process coords in cartesian topology given rank in group
	int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])
	returns the coordinates of the process with the rank "rank" in the cartesian
	communicator "comm"
	*/
	MPI_Cart_coords(grid->comm,grid->my_rank,2,coordinates);
	/* set the coordinate values for the current coordinate */
	grid->my_row=coordinates[0];
	grid->my_col=coordinates[1];

    /* create row communicators */
	free_coords[0]=0;
	free_coords[1]=1; /* row is gonna vary */

	/*
	Partitions a communicator into subgroups which form lower-dimensional cartesian subgrids
	int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)
	*/
	MPI_Cart_sub(grid->comm,free_coords,&(grid->row_comm));
	
    /* create column communicators */
	free_coords[0]=1;
	free_coords[1]=0; /* row is gonna vary */

	/*
	Partitions a communicator into subgroups which form lower-dimensional cartesian subgrids
	int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)
	*/
	MPI_Cart_sub(grid->comm,free_coords,&(grid->col_comm));
	
}

/* normal matrix multiplication method */

void matmul(int **a, int **b, int **c, int size){
	int i,j,k;
	/*
	memory allocation and matrix multiplication algorithm
	*/  
	int **temp = (int**) malloc(size*sizeof(int*));
	for(i=0;i<size;i++)
		*(temp+i)=(int*) malloc(size*sizeof(int));
	for(i=0;i<size;i++){
			for(j=0;j<size;j++){
				temp[i][j]=0;
				for(k=0;k<size;k++){
					temp[i][j]=temp[i][j]+ (a[i][k] * b[k][j]);
				}
			}
	}
	
	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
			c[i][j]+=temp[i][j];
	
}
void transfer_data_from_buff(int *buff,int **a,int buffsize, int row, int col){
	
  	if(buffsize!=row*col){
		printf("transfer_data_from_buf: buffer size does not match matrix size!\n");
		exit(1);
	}
	int count=0, i,j;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			a[i][j]=buff[count];
			count++;
		}
	}
}

void transfer_data_to_buff(int *buff,int **a,int buffsize, int row, int col){
	
  	if(buffsize!=row*col){
		printf("transfer_data_to_buf: buffer size does not match matrix size!");
		exit(1);
	}
	int count=0, i,j;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			buff[count]=a[i][j];
			count++;
		}
	}
}

void Fox(int n,GridInfo *grid,int **a, int **b, int **c)
{
	int **tempa;
	int *buff; /* buffer for Bcast & send_recv */
	int stage; /* stage of fox alg */
	int root;
	int submat_dim; /* = n/q */
	int source;
	int dest;
	int i;
	MPI_Status status;
	
	submat_dim=n/grid->q;
	
	/* Initialize tempa */
	tempa=(int**) malloc(submat_dim*sizeof(int*));
	for(i=0;i<submat_dim;i++)
		*(tempa+i)=(int*) malloc(submat_dim*sizeof(int));
	/* initialize buffer */
	buff=(int*)malloc(submat_dim*submat_dim*sizeof(int));

        /* we are gonna shift the elements of matrix b upwards with the column fixed */
	source = (grid->my_row+1) % grid->q; /* pick the emmediately lower element */
	dest= (grid->my_row+grid->q-1) % grid->q; /* move current element to immediately upper row */
	
	
	for(stage=0;stage<grid->q;stage++){
		root=(grid->my_col+stage)%grid->q;
		if(root==grid->my_col){
			transfer_data_to_buff(buff,a,submat_dim*submat_dim, submat_dim,submat_dim);
			
			/*
			Broadcasts a message from the process with rank "root" to all other processes of the communicator
			int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
               MPI_Comm comm )
			*/
			MPI_Bcast(buff,submat_dim*submat_dim,MPI_INT,root,grid->row_comm);
			transfer_data_from_buff(buff,a,submat_dim*submat_dim, submat_dim,submat_dim);
		
			matmul(a,b,c,submat_dim);
		}else{
			transfer_data_to_buff(buff,tempa,submat_dim*submat_dim, submat_dim,submat_dim);
			
			/*
			Broadcasts a message from the process with rank "root" to all other processes of the communicator
			int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
               MPI_Comm comm )
			*/
			MPI_Bcast(buff,submat_dim*submat_dim,MPI_INT,root,grid->row_comm);
			transfer_data_from_buff(buff,tempa,submat_dim*submat_dim, submat_dim,submat_dim);
			
			matmul(tempa,b,c, submat_dim);
		}
		transfer_data_to_buff(buff,b,submat_dim*submat_dim, submat_dim,submat_dim);
		
		/*
		Sends and receives using a single buffer
		int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, 
                       int dest, int sendtag, int source, int recvtag,
                       MPI_Comm comm, MPI_Status *status)
		*/
		MPI_Sendrecv_replace(buff,submat_dim*submat_dim,MPI_INT,dest,0,source,0,grid->col_comm,&status);
		transfer_data_from_buff(buff,b,submat_dim*submat_dim, submat_dim,submat_dim);
	}

}

/*
A method that deterministically initializes the matrices A and B gives 
the size of a matrix, N. Note that A and B are always the same
For example, for N=2:
A=B= 0 1
	 1 2
*/
void initialiseAB() {
	int i,j;
	for(i=0;i<N;i++){ 
		for(j=0;j<N;j++){ 
			matrixA[i][j]=i+j;
			matrixB[i][j]=i+j;
			printf("%4d", matrixA[i][j]);
			printf("\n");
		}
	}
}


int main(int argc, char *argv[]){

	double starttime, endtime, tick;

    starttime = MPI_Wtime();
    tick = MPI_Wtick();

    clock_t begin = clock();	
	
	int i,j,dim;
	int **localA;
	int **localB;
	int **localC;
	MPI_Init (&argc, &argv);
	
	GridInfo grid;
	/*initialize Grid */

	SetupGrid(&grid);
        /* Initialize matrix A & B */
	initialiseAB();
        /* calculate local matrix dimension */
	dim=N/grid.q;
	/* allocate space for the three matrices */		

	
	localA=(int**) malloc(dim*sizeof(int*));

	localB=(int**) malloc(dim*sizeof(int*));
	
	localC=(int**) malloc(dim*sizeof(int*));
	
	for(i=0;i<dim;i++){
		*(localA+i)=(int*) malloc(dim*sizeof(int));
		*(localB+i)=(int*) malloc(dim*sizeof(int));
		*(localC+i)=(int*) malloc(dim*sizeof(int));
	}


/* Compute local matrices - Ideally the master should do this & pass it onto all the slaves */
/* At the same time initialize localC to all zeros */

	int base_row=grid.my_row*dim;
	int base_col=grid.my_col*dim;

	for(i=base_row;i<base_row+dim;i++){
		for(j=base_col;j<base_col+dim;j++){
		    localA[i-(base_row)][j-(base_col)]=matrixA[i][j];
			localB[i-(base_row)][j-(base_col)]=matrixB[i][j];
			localC[i-(base_row)][j-(base_col)]=0;
		}
	}

	Fox(N,&grid,localA, localB, localC);

/* print results */
	printf("rank=%d, row=%d col=%d\n",grid.my_rank,grid.my_row,grid.my_col);
	for(i=0;i<dim;i++){
		for(j=0;j<dim;j++){
			//printf("localC[%d][%d]=%d ", i,j,localC[i][j]);
			printf("%d ", localC[i][j]);
		}
		printf("\n");
	}
	MPI_Finalize ();
	//exit(0);

	endtime = MPI_Wtime();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Using MPI_Wtime, it took %0.9f seconds\n",endtime-starttime);
    printf("A single MPI tick is %0.9f seconds\n", tick);
    printf("Using C Clock function, CPU time spent is %0.9f seconds\n", time_spent);

    return 0;

}		