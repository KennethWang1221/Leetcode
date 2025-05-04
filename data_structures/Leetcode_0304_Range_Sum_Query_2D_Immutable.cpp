#include <iostream>
#include <vector>

using namespace std;

class NumMatrix {
private:
    vector<vector<int>> sum_;  // 2D prefix sum matrix

public:
    // Constructor to initialize the prefix sum matrix
    NumMatrix(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        // Initialize the prefix sum matrix with an extra row and column (0-indexed)
        sum_ = vector<vector<int>>(rows + 1, vector<int>(cols + 1, 0));
        
        // Populate the prefix sum matrix
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                sum_[r + 1][c + 1] = matrix[r][c] + sum_[r][c + 1] + sum_[r + 1][c] - sum_[r][c];
            }
        }
    }

    // Method to return the sum of the submatrix defined by the top-left and bottom-right coordinates
    int sumRegion(int row1, int col1, int row2, int col2) {
        // Convert to 1-indexed for easier calculation
        row1 += 1;
        col1 += 1;
        row2 += 1;
        col2 += 1;
        
        // Calculate the sum using the inclusion-exclusion principle
        return sum_[row2][col2] - sum_[row1 - 1][col2] - sum_[row2][col1 - 1] + sum_[row1 - 1][col1 - 1];
    }
};

int main() {
    vector<vector<int>> matrix = {
        {3, 0, 1, 4, 2},
        {5, 6, 3, 2, 1},
        {1, 2, 0, 1, 5},
        {4, 1, 0, 1, 7},
        {1, 0, 3, 0, 5}
    };

    NumMatrix numMatrix(matrix);

    cout << numMatrix.sumRegion(2, 1, 4, 3) << endl;  // Expected output: 8
    cout << numMatrix.sumRegion(1, 1, 2, 2) << endl;  // Expected output: 11
    cout << numMatrix.sumRegion(1, 2, 2, 4) << endl;  // Expected output: 12

    return 0;
}

// g++ -std=c++17 Leetcode_0304_Range_Sum_Query_2D_Immutable.cpp -o test