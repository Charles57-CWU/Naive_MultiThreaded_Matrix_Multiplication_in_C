#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>

#define BUFFER 1000

// ===============================Matrix Multiplication Information===============================
// original matrix for powering
long double *original_matrix  = NULL;
// third matrix needed for multiplication
long double *reference_matrix = NULL;
// product matrix for multiplication
long double *product_matrix = NULL;
// job list
int *job_list = NULL;

// doesn't change after dimension and power are set
int dim = 0;
int power = 0;
int random_or_identity = 1;


// for timing matrix powering
int whichTime = 1;
// gettimeofday()
struct timeval gettimeofday_start, gettimeofday_end;
long gettimeofday_total = 0;
// clock()
clock_t clock_start, clock_end, clock_total;
// time()
time_t time_start, time_end, time_total;

// for dialogue
int dialogue = 1;

// ===============================Job & Thread Information===============================

// doesn't change after thread number is set
int number_of_threads = 0;

// increments when a thread goes to sleep because there is no more jobs, resets for each iteration
int threads_sleeping = 0;
pthread_mutex_t thread_sleep_mutex = PTHREAD_MUTEX_INITIALIZER;

// decrements when a job is taken (there is 2 * dim^2 jobs, corresponding to a column+row for dim^2 matrix slots)
// resets for each iteration
int jobs_left = 0;
pthread_mutex_t job_countdown_mutex = PTHREAD_MUTEX_INITIALIZER;

//decrements for each iteration of jobs (powers in this case)
int job_iterations_left = 0;

// communication signals between supervisor and worker
// condition to broadcast to workers
pthread_mutex_t job_iteration_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t job_iteration_cond = PTHREAD_COND_INITIALIZER;
// condition to signal to work supervisor
pthread_cond_t all_jobs_completed_cond = PTHREAD_COND_INITIALIZER;

// just a counter to track threads exiting, not used except in print statement
int threads_exited = 0;
pthread_mutex_t thread_exit_mutex = PTHREAD_MUTEX_INITIALIZER;

// ===============================Thread Pool Information===============================
struct sync_pool {
    pthread_mutex_t *pool_sync_mutex;
    pthread_cond_t *pool_sync_cond;
    pthread_mutex_t *job_iteration_mutex;
    pthread_cond_t *job_iteration_cond;
    int thread_no;
};

// communication signals between pool supervisor and worker
// condition to broadcast to workers
pthread_mutex_t pool_sync_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t pool_sync_cond = PTHREAD_COND_INITIALIZER;
// condition to signal to pool supervisor
pthread_cond_t all_threads_pooled_cond = PTHREAD_COND_INITIALIZER;

// increments as threads fill the pool
int pool_count = 0;

//===================================Matrix Initialization Functions===================================
// generate identity matrix
// generate random matrix
void getMatrix(long double *matrix) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrix[i+j*dim] = rand() % 10;
        }
    }
}


void getIdentityMatrix(long double *matrix) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                matrix[i+j*dim] = 1;
            } else {
                matrix[i+j*dim] = 0;
            }
        }
    }
}

// initialize matrices
void initializeMatrices() {
    original_matrix = malloc(dim * dim * sizeof(long double));
    if (original_matrix == NULL) {
        printf("Memory allocation failure\n");
        exit(1);
    }

    if (random_or_identity == 2) {
        getMatrix(original_matrix);
    } else {
        getIdentityMatrix(original_matrix);
    }

    product_matrix = malloc(dim * dim * sizeof(long double));
    if (product_matrix == NULL) {
        printf("Memory allocation failure\n");
        exit(1);
    }
    memcpy(product_matrix, original_matrix, dim * dim * sizeof(long double));

    reference_matrix = malloc(dim * dim * sizeof(long double));
    if (reference_matrix == NULL) {
        printf("Memory allocation failure\n");
        exit(1);
    }
    memcpy(reference_matrix, original_matrix, dim * dim * sizeof(long double));
}
//===================================Generate Jobs===================================
// specifically built for element by element matrix multiplication
void getJobs() {
    job_list = malloc(2 * dim * dim * sizeof(int));
    if (job_list == NULL) {
        printf("Memory allocation failure\n");
        exit(1);
    }

    // job list sample for a 4x4 matrix, each 'integer' is a job
    // during element matrix multiplication '2 jobs' are taken adjacently relating to col/row pair
    // which is why I stated earlier that there is 2 * dim^2 jobs
    /*
     0 0 0 1 0 2 0 3
     1 0 1 1 1 2 1 3
     2 0 2 1 2 2 2 3
     3 0 3 1 3 2 3 3
     */
    int row_cnt = -1;
    int head_cnt = 0;
    int col_cnt = 0;
    for(int i = 0; i < 2 * dim * dim; i++) {
        if (i % (2*dim) == 0) {
            job_list[i] = head_cnt;
            head_cnt++;
            row_cnt++;
            col_cnt = 0;
        } else if (i % 2 == 0) {
            job_list[i] = row_cnt;
        } else {
            job_list[i] = col_cnt;
            col_cnt++;
        }
    }
}
//===================================Work Functions===================================
// the actual work
void multiplyByElement(int row, int col) {
    product_matrix[col * dim + row] = 0;
    for (int k = 0; k < dim; k++) {
        product_matrix[col * dim + row] += reference_matrix[k * dim + row] * original_matrix[col * dim + k];
    }
}

// work supervisor, this moves workers to the next job iteration
void* workSupervisorFunc(void *arg) {
    struct sync_pool *supervisor_info = arg;

    while (job_iterations_left > 0) {
        pthread_mutex_lock(supervisor_info->job_iteration_mutex);
        while (threads_sleeping < number_of_threads) {
            pthread_cond_wait(&all_jobs_completed_cond, supervisor_info->job_iteration_mutex);
        }

        printf("All elements calculated. Work supervisor thread preparing for next job iteration\n");
        // this is all that is needed to prep for matrix powering
        memcpy(reference_matrix, product_matrix, dim * dim * sizeof(long double));

        // reset job counters
        pthread_mutex_lock(&job_countdown_mutex);
        jobs_left = 2*dim*dim - 1;
        pthread_mutex_unlock(&job_countdown_mutex);

        pthread_mutex_lock(&thread_sleep_mutex);
        threads_sleeping = 0;
        pthread_mutex_unlock(&thread_sleep_mutex);

        // decrement job iterations
        job_iterations_left--;

        if(job_iterations_left > 0) {
            printf("Work supervisor says to do some new jobs\n");
            printf("===================================Worker threads now calculating A^%d=================================== \n", -(job_iterations_left - power - 1));
        } else {
            printf("===================================Finished Calculation===================================\n");
            printf("No more job iterations to assign.\n");
            printf("Work supervisor says to go home\n");
            printf("Work supervisor thread has exited the program\n");
        }
        // wake up workers
        pthread_cond_broadcast(supervisor_info->job_iteration_cond);
        pthread_mutex_unlock(supervisor_info->job_iteration_mutex);

    }

    return NULL;
}

// assigns work to a thread
void threadWorkToDo(void*arg){
    struct sync_pool *worker_info = arg;
    int row = 0;
    int col = 0;

    while (job_iterations_left > 0) {
        if(dialogue == 2) {
            printf("Thread %d is now working\n", worker_info->thread_no);
        }
        while (jobs_left >= 0) {
            pthread_mutex_lock(&job_countdown_mutex);
            // if statement needed in case another thread has entered loop but waiting on mutex
            // and last job is taken
            if (jobs_left > 0) {
                row = job_list[jobs_left];
                jobs_left--;
                col = job_list[jobs_left];
                jobs_left--;
                pthread_mutex_unlock(&job_countdown_mutex);
            } else {
                pthread_mutex_unlock(&job_countdown_mutex);
            }
            // if statement need in case another thread has entered loop but waiting on mutex
            // and last job is taken
            if (jobs_left > -2) {
                multiplyByElement(col, row);

                if(dialogue == 2) {
                    printf("Thread %d has calculated element %d, %d\n", worker_info->thread_no, col, row);
                }
            }
        }
        //begin thread sleeping
        pthread_mutex_lock(worker_info->job_iteration_mutex);

        pthread_mutex_lock(&thread_sleep_mutex);
        threads_sleeping++;
        pthread_mutex_unlock(&thread_sleep_mutex);

        if(dialogue == 2) {
            printf("Thread %d is now sleeping\n", worker_info->thread_no);
        }
        // last thread signals that all are sleeping
        if (threads_sleeping >= number_of_threads) {
            pthread_cond_signal(&all_jobs_completed_cond);
        }
        // thread sleeping aka waiting for work supervisor to move to next iteration
        pthread_cond_wait(worker_info->job_iteration_cond, worker_info->job_iteration_mutex);
        pthread_mutex_unlock(worker_info->job_iteration_mutex);
    }

    // exit dialogue
    if(dialogue == 2) {
        printf("Thread %d has exited the program\n", worker_info->thread_no);
    }
    pthread_mutex_lock(&thread_exit_mutex);
    threads_exited++;
    pthread_mutex_unlock(&thread_exit_mutex);
}

//===================================Thread Pool Functions===================================
// pool the workers
void  *threadPoolFunction(void* arg) {
    struct sync_pool *worker_info = arg;
    int threadID = worker_info->thread_no;

    pthread_mutex_lock(worker_info->pool_sync_mutex);

    if(dialogue == 2) {
        printf("Thread %d has been created\n", threadID);
    }
    pool_count++;

    // last thread signals that all are pooled
    if (pool_count >= number_of_threads) {
        pthread_cond_signal(&all_threads_pooled_cond);
    }

    // thread sleeping aka waiting for pool supervisor to release everyone
    pthread_cond_wait(worker_info->pool_sync_cond, worker_info->pool_sync_mutex);
    pthread_mutex_unlock(worker_info->pool_sync_mutex);

    //threads can get some work here
    threadWorkToDo(arg);

    return NULL;
}

// supervisor function to make sure the workers have pooled and release them to work
void *threadPoolSupervisorFunc(void *arg) {
    struct sync_pool *supervisor_info = arg;

    pthread_mutex_lock(supervisor_info->pool_sync_mutex);
    // wait for threads to pool
    while (pool_count < number_of_threads) {
        pthread_cond_wait(&all_threads_pooled_cond, supervisor_info->pool_sync_mutex);
    }

    printf("All threads pooled. Pool supervisor thread says to begin working\n");
    printf("===================================Worker threads now calculating A^%d=================================== \n",-(job_iterations_left - power - 1));

    //all pooled signal the threads to begin working
    pthread_cond_broadcast(supervisor_info->pool_sync_cond);
    pthread_mutex_unlock(supervisor_info->pool_sync_mutex);

    if(whichTime == 1) {
        time_start = time(NULL);
    } else if (whichTime == 2) {
        clock_start = clock();
    } else {
        gettimeofday(&gettimeofday_start, NULL);
    }

    // easy job for this one
    printf("Pool supervisor thread has exited the program\n");
    return NULL;
}
//===================================Assistant Functions===================================
// check identity matrix
void checkIdentity(long double *matrix) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                if (matrix[i+j*dim] == 1) {
                    continue;
                } else {
                    printf("Identity matrix is wrong.\n");
                    return;
                }
            } else {
                if (matrix[i+j*dim] == 0) {
                    continue;
                } else {
                    printf("Identity matrix is wrong.\n");
                    return;
                }
            }
        }
    }
    printf("Identity matrix is correct.\n");
}

// print matrix
void printMatrix(long double* matrix) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%Lf ", matrix[i+j*dim]);
        }
        printf("\n");
    }
}

//===================================Input from Keyboard===================================
void getInputs() {
    //get power of matrix from keyboard
    char line[BUFFER + 1] = {0x0};
    char *check;

    //get dimension
    printf("Please input via keyboard the integer dimension M (1000 >= M >= 2).\n");
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    //check input is correct
    dim = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect dimension inputted.\n");
        printf("Dimension must be an integer between 2 and 1000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (dim < 2 || dim > 1000) {
        printf("Incorrect dimension inputted.\n");
        printf("Dimension must be an integer between 2 and 1000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    }


    // get power
    printf("Please input via keyboard the integer power N (5000 >= N >= 1).\n");
    printf("Warning: Large powers will result in overflow unless you choose the identity matrix option\n");
    printf("The long double data type on this machine has a max value of %LE\n", LDBL_MAX);
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    //check input is correct
    power = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect power inputted.\n");
        printf("Power must be an integer between 1 and 5000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (power < 1 || power > 5000) {
        printf("Incorrect power inputted.\n");
        printf("Power must be an integer between 1 and 5000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    }


    // get number of threads
    printf("Please input via keyboard the number of threads (1000 >= x >= 1).\n");
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    //check input is correct
    number_of_threads = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect power inputted.\n");
        printf("Number of threads must be an integer between 1 and 1000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (number_of_threads < 1 || number_of_threads > 1000) {
        printf("Incorrect power inputted.\n");
        printf("Number of threads must be an integer between 1 and 1000 inclusive\n");
        printf("Exiting Program.\n");
        exit(1);
    }


    //get identity or random matrix
    printf("Please type 1 for identity matrix or 2 for random matrix\n");
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    *check;
    random_or_identity = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (random_or_identity != 2 && random_or_identity != 1) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2\n");
        printf("Exiting Program.\n");
        exit(1);
    }


    //get time choice
    printf("Please type 1 for time() or 2 for clock() or 3 for gettimeofday()\n");
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    *check;
    whichTime = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2 or 3\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (whichTime != 2 && whichTime != 1 && whichTime != 3) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2 or 3\n");
        printf("Exiting Program.\n");
        exit(1);
    }


    //get dialogue choice
    printf("Please type 1 for minimal dialogue or 2 for extended dialogue\n");
    printf("Option 2 is not recommended for dimensions over a hundred\n");
    fgets(line, BUFFER, stdin);
    if (strcmp(line, "\n") == 0) {
        printf("No input detected.\n");
        printf("Exiting Program.\n");
        exit(1);
    }
    line[strcspn(line, "\r\n")] = 0;

    *check;
    dialogue = strtol(line, &check, 10);
    if (*check) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2\n");
        printf("Exiting Program.\n");
        exit(1);
    } else if (dialogue != 2 && dialogue != 1) {
        printf("Incorrect choice inputted.\n");
        printf("Choice must be 1 or 2\n");
        printf("Exiting Program.\n");
        exit(1);
    }
}

int main() {
    srand(15);
    getInputs();

    printf("You have chosen the following:\n\n");
    if (random_or_identity == 1) {
        printf("Dimension: %d\nPower: %d\nThreads: %d\nType: Identity\n", dim, power, number_of_threads);
    } else {
        printf("Dimension: %d\nPower: %d\nThreads: %d\nType: Random Values\n", dim, power, number_of_threads);
    }

    if(whichTime == 1) {
        printf("Time Function: time()\n");
    } else if (whichTime == 2) {
        printf("Time Function: clock()\n");
    } else {
        printf("Time Function: gettimeofday()\n");
    }

    if (dialogue == 1) {
        printf("Dialogue: Minimal\n");
    } else {
        printf("Dialogue: Extended\n");
    }

    // dim - 1 needed for matrix multiplication as array indices start at 0
    // each thread takes 2 jobs at a time the correspond to a column and row index of the product matrix
    // jobs_left = 0, and jobs_left = 1 would be element 0,0
    // jobs_left = 2, and jobs_left = 3 would be element 0,1
    // .....
    // jobs_left = 2*dim^2 - 1, and jobs_left 2*dim^2 - 2 would be element dim-1, dim-1
    jobs_left = 2*dim*dim - 1;
    job_iterations_left = power -1;

    printf("Multiplication will begin in 5 seconds\n");
    sleep(5);
    printf("\nGenerating Matrix Values\n");

    initializeMatrices();
    //printMatrix(original_matrix);
    getJobs();

    // create worker threads
    pthread_t worker[number_of_threads];
    struct sync_pool info[number_of_threads];

    for (int i = 0; i < number_of_threads; i++) {

        struct sync_pool *worker_arg = &info[i];
        worker_arg->pool_sync_mutex = &pool_sync_mutex;
        worker_arg->pool_sync_cond = &pool_sync_cond;
        worker_arg->job_iteration_mutex = &job_iteration_mutex;
        worker_arg->job_iteration_cond = &job_iteration_cond;
        worker_arg->thread_no = i;

        pthread_create(&worker[i], NULL, threadPoolFunction, worker_arg);
    }

    // create two supervisor threads
    struct sync_pool supervisor_arg = {&pool_sync_mutex, &pool_sync_cond, NULL, NULL, number_of_threads};
    pthread_t pool_supervisor;
    pthread_create(&pool_supervisor, NULL, threadPoolSupervisorFunc, &supervisor_arg);

    struct sync_pool work_sup_arg = {NULL, NULL, &job_iteration_mutex, &job_iteration_cond, number_of_threads};
    pthread_t work_supervisor;
    pthread_create(&work_supervisor, NULL, workSupervisorFunc, &work_sup_arg);

    // join threads
    pthread_join(pool_supervisor, NULL);

    for (int i = 0; i < number_of_threads; i++) {
        pthread_join(worker[i], NULL);
    }

    pthread_join(work_supervisor, NULL);


// exit dialogue
    printf("%d worker threads have exited\n", threads_exited);
    if (random_or_identity == 1) {
        printf("Checking identity matrix\n");
        checkIdentity(product_matrix);
    }
    //printMatrix(product_matrix);

    // calculate the time in ms
    if(whichTime == 1) {
        time_end = time(NULL);
        time_total = time_end - time_start;
        printf("Total time for %d threads to calculate A^%d for %d x %d is %ld ms\n", number_of_threads, power, dim, dim, time_total*1000);
    } else if (whichTime == 2) {
        clock_end = clock();
        clock_total = clock_end - clock_start;
        printf("Total time for %d threads to calculate A^%d for %d x %d is %ld ms\n", number_of_threads, power, dim, dim, clock_total / CLOCKS_PER_SEC *1000);
    } else {
        gettimeofday(&gettimeofday_end, NULL);
        gettimeofday_total = (gettimeofday_end.tv_sec * 1000000 + gettimeofday_end.tv_usec) - (gettimeofday_start.tv_sec * 1000000 + gettimeofday_start.tv_usec);
        gettimeofday_total = gettimeofday_total / 1000;
        printf("Total time for %d threads to calculate A^%d for %d x %d is %ld ms\n", number_of_threads, power, dim, dim, gettimeofday_total);
    }

    // free everything
    free(original_matrix);
    free(reference_matrix);
    free(product_matrix);
    free(job_list);

    printf("Program Finished\n\n");
    return 0;
}


