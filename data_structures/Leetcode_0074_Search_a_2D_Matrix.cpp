#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        int top = 0, bot = rows - 1;
        
        // Perform binary search on the rows
        while (top <= bot) {
            int mid = (top + bot) / 2;
            if (target > matrix[mid][cols - 1]) {
                top = mid + 1;
            } else if (target < matrix[mid][0]) {
                bot = mid - 1;
            } else {
                break;
            }
        }
        
        // If no row was found, return false
        if (top > bot) return false;
        
        int row = (top + bot) / 2;  // The row where the target could be
        vector<int> select = matrix[row];
        
        int start = 0, end = cols - 1;
        
        // Perform binary search on the selected row
        while (start <= end) {
            int mid = (start + end) / 2;
            if (target > select[mid]) {
                start = mid + 1;
            } else if (target < select[mid]) {
                end = mid - 1;
            } else {
                return true;  // Target found
            }
        }
        
        return false;  // Target not found
    }
};

int main() {
    Solution solution;
    
    vector<vector<int>> matrix = {{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 60}};
    int target = 3;
    
    bool result = solution.searchMatrix(matrix, target);
    cout << "Target " << target << " found: " << result << endl;  // Expected output: true
    
    target = 2;
    result = solution.searchMatrix(matrix, target);
    cout << "Target " << target << " found: " << result << endl;  // Expected output: false
    
    return 0;
}

// g++ -std=c++17 Leetcode_0074_Search_a_2D_Matrix.cpp -o test