#include <iostream>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();

        // DP table with extra row and column initialized to infinity
        vector<vector<int>> res(rows + 1, vector<int>(cols + 1, INT_MAX));
        // res[rows - 1][cols] = 0; // Base case, use this or next line , both ok.
        res[rows][cols-1] = 0;  // Base case, 
        for (int r = rows - 1; r >= 0; --r) {
            for (int c = cols - 1; c >= 0; --c) {
                res[r][c] = grid[r][c] + min(res[r + 1][c], res[r][c + 1]);
            }
        }

        return res[0][0];
    }
};

// Test Case
int main() {
    Solution sol;
    vector<vector<int>> grid = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
    cout << sol.minPathSum(grid) << endl; // Output: 7
    return 0;
}

// g++ -std=c++17 Leetcode_0064_Minimum_Path_Sum.cpp -o test 