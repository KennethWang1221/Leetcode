// softmax.c
#include <stdio.h>
#include <math.h>

// Function to compute the maximum value in an array
double find_max(const double* input, int length) {
    double max = input[0];
    for(int i = 1; i < length; i++) {
        if(input[i] > max) {
            max = input[i];
        }
    }
    return max;
}

// Function to compute the Softmax of an input array
void softmax(const double* input, double* output, int length) {
    double max = find_max(input, length); // For numerical stability
    double sum = 0.0;

    // Compute the exponentials and sum them
    for(int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    // Normalize the exponentials to get probabilities
    for(int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Function to print an array
void print_array(const double* array, int length) {
    printf("[");
    for(int i = 0; i < length; i++) {
        printf("%.6f", array[i]);
        if(i < length - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
    // Sample input vector
    double input[] = {2.0, 1.0, 0.1};
    int length = sizeof(input) / sizeof(input[0]);
    double output[length];

    // Compute Softmax
    softmax(input, output, length);

    // Print input and output
    printf("Input: ");
    print_array(input, length);

    printf("Softmax Output: ");
    print_array(output, length);

    return 0;
}
