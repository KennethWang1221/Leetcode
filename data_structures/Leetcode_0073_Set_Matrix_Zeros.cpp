#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int ROWS = matrix.size();
        int COLS = matrix[0].size();
        bool rowZero = false;

        // Determine which rows and columns need to be zero
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (matrix[r][c] == 0) {
                    matrix[0][c] = 0;  // Mark the column
                    if (r > 0) {
                        matrix[r][0] = 0;  // Mark the row
                    } else {
                        rowZero = true;  // Special case for the first row
                    }
                }
            }
        }

        // Use the markers to set appropriate elements to zero
        for (int r = 1; r < ROWS; ++r) {
            for (int c = 1; c < COLS; ++c) {
                if (matrix[0][c] == 0 || matrix[r][0] == 0) {
                    matrix[r][c] = 0;
                }
            }
        }

        // Set the first column to zero if needed
        if (matrix[0][0] == 0) {
            for (int r = 0; r < ROWS; ++r) {
                matrix[r][0] = 0;
            }
        }

        // Set the first row to zero if needed
        if (rowZero) {
            for (int c = 0; c < COLS; ++c) {
                matrix[0][c] = 0;
            }
        }
    }
};

int main() {
    Solution solution;

    // Example usage
    vector<vector<int>> matrix = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    
    solution.setZeroes(matrix);

    // Print the matrix after setting zeros
    cout << "Matrix after setting zeroes:" << endl;
    for (const auto& row : matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }

    return 0;
}

// g++ -std=c++17 Leetcode_0073_Set_Matrix_Zeros.cpp -o test && ./test