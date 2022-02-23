#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <caliper/cali.h>
#include <mpi.h>

int interval = 10;
 
void foo (int _interval) {
 
        printf("foo function starts here !!!\n");
 
        int j;
        for (j=0; j<=_interval; j++) {
                printf("foo counter value is: %d\n",j);
        }
 
}
int main(int argc, char** argv) 
{
        MPI_Init(&argc, &argv);

        CALI_MARK_FUNCTION_BEGIN;
 
        cali_begin_byname("ldms");
        // Mark the "intialization" phase
        CALI_MARK_BEGIN("initialization");
 
        int i = 0;
        printf("%d\n", argc);
        if (argc == 2) {
                for (; argv[1][i] != 0; i++)
                {
                        if (!isdigit(argv[1][i])) {
                                printf("Error: a number is required!!!\n");
                                exit(1);
                        }
                }
                interval = atoi(argv[1]);
                printf("The interval set to %d \n", interval);
        }
 
        // doing nothing!
        for (i=0; i<=interval; i++);
 
 
 
        CALI_MARK_LOOP_BEGIN(mainloop, "mainloop");
 
        for (i=0; i<=interval; i++) {
//              CALI_MARK_ITERATION_BEGIN(mainloop, i);
                printf("main counter value is: %d\n",i);
//              CALI_MARK_ITERATION_END(mainloop);
        }
 
        CALI_MARK_LOOP_END(mainloop);
 
        foo(interval);

        MPI_Barrier(MPI_COMM_WORLD);
 
        CALI_MARK_END("initialization");
        cali_end_byname("ldms");
 
        CALI_MARK_FUNCTION_END;

        MPI_Finalize();
}
 