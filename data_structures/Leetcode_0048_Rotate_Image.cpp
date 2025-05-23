#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        int left = 0, right = n - 1;

        // Rotate layer by layer
        while (left < right) {
            // Rotate the current layer
            for (int i = 0; i < right - left; ++i) {
                int top = left, bottom = right;

                // Save the top-left element
                int topleft = matrix[top][left + i];

                // Move bottom-left to top-left
                matrix[top][left + i] = matrix[bottom - i][left];

                // Move bottom-right to bottom-left
                matrix[bottom - i][left] = matrix[bottom][right - i];

                // Move top-right to bottom-right
                matrix[bottom][right - i] = matrix[top + i][right];

                // Move topleft to top-right
                matrix[top + i][right] = topleft;
            }
            left++;
            right--;
        }
    }
};

int main() {
    Solution solution;

    // Test case
    vector<vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    solution.rotate(matrix);

    // Print the rotated matrix
    cout << "Rotated Matrix:" << endl;
    for (const auto& row : matrix) {
        for (int num : row) {
            cout << num << " ";
        }
        cout << endl;
    }

    return 0;
}
// g++ -std=c++11 Leetcode_0048_Rotate_Image.cpp -o test && ./test