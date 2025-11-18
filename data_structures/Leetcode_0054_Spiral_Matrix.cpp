#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> spiralOrder_pointer(vector<vector<int>>* matrix) {
        int rows = matrix->size();
        int cols = (*matrix)[0].size();
        
        vector<int> res;
        int top = 0, left = 0, bottom = rows - 1, right = cols - 1;

        while (left <= right && top <= bottom) {

            // Traverse from left to right along the top row
            for (int c = left; c <= right; ++c) {
                res.push_back((*matrix)[top][c]);
            }
            top++;

            // Traverse from top to bottom along the right column
            for (int r = top; r <= bottom; ++r) {
                res.push_back((*matrix)[r][right]);
            }
            right--;

            if (left > right || top > bottom) {
                break;
            }

            // Traverse from right to left along the bottom row
            for (int c = right; c >= left; --c) {
                res.push_back((*matrix)[bottom][c]);
            }
            bottom--;

            // Traverse from bottom to top along the left column
            for (int r = bottom; r >= top; --r) {
                res.push_back((*matrix)[r][left]);
            }
            left++;
        }

        return res;
    }

    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        vector<int> res;
        int top = 0, left = 0, bottom = rows - 1, right = cols - 1;

        while (left <= right && top <= bottom) {

            // Traverse from left to right along the top row
            for (int c = left; c <= right; ++c) {
                res.push_back(matrix[top][c]);
            }
            top++;

            // Traverse from top to bottom along the right column
            for (int r = top; r <= bottom; ++r) {
                res.push_back(matrix[r][right]);
            }
            right--;

            if (left > right || top > bottom) {
                break;
            }

            // Traverse from right to left along the bottom row
            for (int c = right; c >= left; --c) {
                res.push_back(matrix[bottom][c]);
            }
            bottom--;

            // Traverse from bottom to top along the left column
            for (int r = bottom; r >= top; --r) {
                res.push_back(matrix[r][left]);
            }
            left++;
        }

        return res;
    }
};

int main() {
    Solution solution;

    // Test case
    vector<vector<int>> matrix = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    vector<vector<int>> matrix2 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    vector<int> result = solution.spiralOrder(matrix);
    vector<int> result2 = solution.spiralOrder_pointer(&matrix2);
    // Print the result
    cout << "Spiral Order: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    for (int num : result2) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
// g++ -std=c++17 -g Leetcode_0054_Spiral_Matrix.cpp -o test