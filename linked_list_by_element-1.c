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
// the node structure
struct Node {
    long double value;
    struct Node *down;
    struct Node *right;
};

struct Node *original_head = NULL;
struct Node *original_matrix_col= NULL;
pthread_mutex_t original_matrix_col_mutex = PTHREAD_MUTEX_INITIALIZER;


struct Node *product_head = NULL;
struct Node *product_matrix_curr = NULL;
pthread_mutex_t product_matrix_curr_mutex = PTHREAD_MUTEX_INITIALIZER;
struct Node *product_matrix_row = NULL;
pthread_mutex_t product_matrix_row_mutex = PTHREAD_MUTEX_INITIALIZER;

struct Node *reference_head = NULL;
struct Node *reference_matrix_row = NULL;
pthread_mutex_t reference_matrix_row_mutex = PTHREAD_MUTEX_INITIALIZER;

// doesn't change
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
int row_index = 0;
int col_index = 0;
// ===============================Job & Thread Information===============================

// doesn't change
int number_of_threads = 0;
int total_jobs = 0;

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
// code modified from https://www.geeksforgeeks.org/construct-a-linked-list-from-2d-matrix-iterative-approach/


// add new node to identity matrix
struct Node* identityNode(int i, int j) {
    struct Node* temp = malloc(sizeof(struct Node));
    if (i == j) {
        temp->value = 1;
    } else {
        temp->value = 0;
    }
    temp->right = temp->down = NULL;
    return temp;
}

// construct identity matrix
struct Node* constructLinkedIdenMatrix(struct Node *mainhead) {

    //struct Node* mainhead = NULL;

    struct Node* head[dim];
    struct Node *righttemp, *newptr;

    for (int i = 0; i < dim; i++) {
        head[i] = NULL;
        for (int j = 0; j < dim; j++) {
            //newptr = newNode();
            newptr = identityNode(i,j);
            if (!mainhead) {
                mainhead = newptr;
            }
            if (!head[i]) {
                head[i] = newptr;
            } else {
                righttemp->right = newptr;
            }
            righttemp = newptr;
        }
    }

    for (int i = 0; i < dim - 1; i++) {
        struct Node *temp1 = head[i], *temp2 = head[i + 1];
        while (temp1 && temp2) {
            temp1->down = temp2;
            temp1 = temp1->right;
            temp2 = temp2->right;
        }
    }
    return mainhead;
}

// add new node to random matrix
struct Node* newNode() {
    struct Node* temp = malloc(sizeof(struct Node));
    if(temp == NULL) {
        printf("Memory allocations failure\n");
        exit(1);
    }
    temp->value = rand()%100;
    //printf("%Lf ", temp->value);
    temp->right = temp->down = NULL;
    return temp;
}

// construct random matrix
struct Node* constructLinkedMatrix(struct Node* mainhead) {

    struct Node* head[dim];
    struct Node *righttemp, *newptr;

    for (int i = 0; i < dim; i++) {
        head[i] = NULL;
        for (int j = 0; j < dim; j++) {
            newptr = newNode();
            //newptr = identityNode(i,j);
            if (!mainhead) {
                mainhead = newptr;
            }
            if (!head[i]) {
                head[i] = newptr;
            } else {
                righttemp->right = newptr;
            }
            righttemp = newptr;
        }
    }

    for (int i = 0; i < dim - 1; i++) {
        struct Node *temp1 = head[i], *temp2 = head[i + 1];
        while (temp1 && temp2) {
            temp1->down = temp2;
            temp1 = temp1->right;
            temp2 = temp2->right;
        }
    }
    return mainhead;
}

// new node function for copy matrix
struct Node* newCopyNode(struct Node *toCopy) {
    struct Node* temp = malloc(sizeof(struct Node));
    if(temp == NULL) {
        printf("Memory allocations failure\n");
        exit(1);
    }
    temp->value = toCopy->value;
    //printf("%Lf ", temp->value);
    temp->right = temp->down = NULL;
    return temp;
}

// copy function for copying a linked list
struct Node* copy(struct Node *toCopy) {

    struct Node* mainhead = NULL;
    struct Node* toCopyDown = toCopy;
    struct Node* head[dim];
    struct Node *righttemp, *newptr;

    for (int i = 0; i < dim; i++) {
        head[i] = NULL;
        for (int j = 0; j < dim; j++) {
            newptr = newCopyNode(toCopy);
            toCopy = toCopy->right;
            if (!mainhead) {
                mainhead = newptr;
            }
            if (!head[i]) {
                head[i] = newptr;
            } else {
                righttemp->right = newptr;
            }
            righttemp = newptr;
        }
        toCopyDown = toCopyDown->down;
        toCopy = toCopyDown;
    }

    for (int i = 0; i < dim - 1; i++) {
        struct Node *temp1 = head[i], *temp2 = head[i + 1];
        while (temp1 && temp2) {
            temp1->down = temp2;
            temp1 = temp1->right;
            temp2 = temp2->right;
        }
    }
    return mainhead;
}


//===================================Work Functions===================================
// the actual work
void freeNodeStructure(struct Node *nodes_to_free);

void multiplyByElement(struct Node *multiply_reference, struct Node *multiply_original, struct Node *multiply_product) {
    multiply_product->value = 0;
    for (int k = 0; k < dim; k++) {
        multiply_product->value += multiply_reference->value * multiply_original->value;
        multiply_reference = multiply_reference->right;
        multiply_original = multiply_original->down;
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
        freeNodeStructure(reference_head);
	reference_head = copy(product_head);


        pthread_mutex_lock(&original_matrix_col_mutex);
        original_matrix_col = original_head;
        pthread_mutex_unlock(&original_matrix_col_mutex);

        pthread_mutex_lock(&product_matrix_curr_mutex);
        product_matrix_curr = product_head;
        pthread_mutex_unlock(&product_matrix_curr_mutex);

        pthread_mutex_lock(&product_matrix_row_mutex);
        product_matrix_row = product_head;
        pthread_mutex_unlock(&product_matrix_row_mutex);

        pthread_mutex_lock(&reference_matrix_row_mutex);
        reference_matrix_row = reference_head;
        pthread_mutex_unlock(&reference_matrix_row_mutex);

        // reset job counters
        pthread_mutex_lock(&job_countdown_mutex);
        jobs_left = total_jobs;
        if(dialogue == 2) {
            col_index = 0;
            row_index = 0;
        }
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

    struct Node *multiply_reference;
    struct Node *multiply_original;
    struct Node *multiply_product;

    while (job_iterations_left > 0) {
        if(dialogue == 2) {
            printf("Thread %d is now working\n", worker_info->thread_no);
        }
        while (jobs_left > 0) {
            pthread_mutex_lock(&job_countdown_mutex);
            if (jobs_left == total_jobs && jobs_left != 0) {

                jobs_left--;
                multiply_reference = reference_head;
                multiply_original = original_head;
                multiply_product = product_head;

                if(dialogue == 2) {
                    col_index++;
                }
                pthread_mutex_unlock(&job_countdown_mutex);

                multiplyByElement(multiply_reference, multiply_original, multiply_product);

                if(dialogue == 2) {
                    printf("Thread %d is calculating element %d, %d\n", worker_info->thread_no, col_index - 1, row_index);
                }
            } else if (jobs_left % dim == 0 && jobs_left != 0) {
                jobs_left--;

                if(dialogue == 2) {
                    row_index++;
                    col_index = 0;
                }

                // move down to next row in product matrix then update current product element
                pthread_mutex_lock(&product_matrix_row_mutex);
                product_matrix_row = product_matrix_row->down;

                pthread_mutex_lock(&product_matrix_curr_mutex);
                product_matrix_curr = product_matrix_row;
                multiply_product = product_matrix_curr;
                pthread_mutex_unlock(&product_matrix_curr_mutex);

                pthread_mutex_unlock(&product_matrix_row_mutex);
                // move down to next row in reference matrix
                pthread_mutex_lock(&reference_matrix_row_mutex);
                reference_matrix_row = reference_matrix_row->down;
                multiply_reference = reference_matrix_row;
                pthread_mutex_unlock(&reference_matrix_row_mutex);
                // reset original matrix column to the left
                pthread_mutex_lock(&original_matrix_col_mutex);
                original_matrix_col = original_head;
                multiply_original = original_matrix_col;
                pthread_mutex_unlock(&original_matrix_col_mutex);

                if(dialogue == 2) {
                    printf("Thread %d is calculating element %d, %d\n", worker_info->thread_no, col_index, row_index);
                    col_index++;
                }

                pthread_mutex_unlock(&job_countdown_mutex);
                multiplyByElement(multiply_reference, multiply_original, multiply_product);
            } else if (jobs_left != 0) {
                jobs_left--;

                // move original matrix column to the right
                pthread_mutex_lock(&reference_matrix_row_mutex);
                multiply_reference = reference_matrix_row;
                pthread_mutex_unlock(&reference_matrix_row_mutex);

                pthread_mutex_lock(&original_matrix_col_mutex);
                original_matrix_col = original_matrix_col->right;
                multiply_original = original_matrix_col;
                pthread_mutex_unlock(&original_matrix_col_mutex);

                // move current product element to the right
                pthread_mutex_lock(&product_matrix_curr_mutex);
                product_matrix_curr = product_matrix_curr->right;
                multiply_product = product_matrix_curr;
                pthread_mutex_unlock(&product_matrix_curr_mutex);
                if(dialogue == 2) {
                    printf("Thread %d is calculating element %d, %d\n", worker_info->thread_no, col_index, row_index);
                    col_index++;
                }
                pthread_mutex_unlock(&job_countdown_mutex);
                multiplyByElement(multiply_reference, multiply_original, multiply_product);
            } else {
                pthread_mutex_unlock(&job_countdown_mutex);
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


    pthread_mutex_lock(worker_info->pool_sync_mutex);
    if(dialogue == 2) {
        printf("Thread %d has been created\n", worker_info->thread_no);
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
//===================================Assistant Functions===================================
void display(struct Node* head) {
    struct Node *rp, *dp = head;

    // loop until the down pointer is not NULL
    while (dp) {
        rp = dp;

        // loop until the right pointer is not NULL
        while (rp) {
            printf("%Lf ", rp->value);
            rp = rp->right;
        }
        printf("\n");
        dp = dp->down;
    }
}


void checkIdentity(struct Node* head) {
    struct Node *rp;
    struct Node *dp = head;

    rp = dp;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                if (rp->value == 1) {
                    rp = rp->right;
                } else {
                    printf("Identity matrix is wrong.\n");
                    return;
                }
            } else {
                if(rp->value == 0) {
                    rp = rp->right;
                } else {
                    printf("Identity matrix is wrong.\n");
                    return;
                }
            }
        }
        dp = dp->down;
        rp = dp;
    }
    printf("Identity matrix is correct\n");
}

void freeNodeStructure(struct Node *nodes_to_free) {
    struct Node* free_node;
    struct Node* row_deleter = nodes_to_free;

    struct Node* nextNode = row_deleter;
    struct Node* next_row = row_deleter->down;


    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            free_node = nextNode;
            if(j != dim - 1) {
                nextNode = nextNode->right;
            }
            free(free_node);
        }
        nextNode = next_row;
        if (i != dim-1) {
            next_row = next_row->down;
        }
    }

}


int main() {

    srand(9);
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

    total_jobs = dim * dim;
    jobs_left = total_jobs;
    job_iterations_left = power - 1;

    //generate matrix
    printf("Multiplication will begin in 5 seconds\n");
    sleep(5);
    printf("\nGenerating Matrix Values\n");

    if (random_or_identity == 2) {
        original_head = constructLinkedMatrix(original_head);
    } else {
        original_head = constructLinkedIdenMatrix(original_head);
    }
    //display(original_head);

    // initialize nodes for keeping track of position during multiplications
    product_head = copy(original_head);
    reference_head = copy(original_head);

    original_matrix_col = original_head;
    product_matrix_curr = product_head;
    product_matrix_row = product_head;
    reference_matrix_row = reference_head;

    printf("\nNow solving A^%d for a square matrix of dimension %d.\n", power, dim);

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

    // exit dialogue
    printf("%d worker threads have exited\n", threads_exited);

    if (random_or_identity == 1) {
        printf("\nChecking identity matrix\n");
        checkIdentity(product_head);
    }

    //display(product_head);

    // free everything
    freeNodeStructure(original_head);
    freeNodeStructure(product_head);
    freeNodeStructure(reference_head);

    printf("Program Finished.\n");
    return 0;
}
