#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> transpose_matrix(const vector<vector<int>>& matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        
        // Initialize the transposed matrix with empty vectors
        vector<vector<int>> transposed(col, vector<int>(row));

        // Iterate through rows and columns to fill the transposed matrix
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                transposed[c][r] = matrix[r][c];
            }
        }
        
        return transposed;
    }
};

int main() {
    Solution solution;

    // Example usage
    vector<vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    };

    // Compute the transposed matrix
    vector<vector<int>> transposed_matrix = solution.transpose_matrix(matrix);

    // Print the original matrix
    cout << "Original Matrix:" << endl;
    for (const auto& row : matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }

    // Print the transposed matrix
    cout << "\nTransposed Matrix:" << endl;
    for (const auto& row : transposed_matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }

    return 0;
}
// g++ -std=c++11 Leetcode_0867_Transpose_Matrix.cpp -o test && ./test