#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    unordered_map<int, int> dp; // Use hash of (r, c) as key -> LIP value

    int dfs(int r, int c, int prevVal, vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        // Boundary and increasing path check
        if (r < 0 || c < 0 || r >= rows || c >= cols || matrix[r][c] <= prevVal) {
            return 0;
        }

        // Create a unique key for (r,c)
        int key = r * 1000 + c;

        if (dp.count(key)) {
            return dp[key];
        }

        int res = 1; // At least itself

        for (auto& dir : directions) {
            int nr = r + dir[0];
            int nc = c + dir[1];
            res = max(res, 1 + dfs(nr, nc, matrix[r][c], matrix));
        }

        dp[key] = res;
        return res;
    }

    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        int maxPath = 0;

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                maxPath = max(maxPath, dfs(r, c, -1, matrix)); // Start with -1 as prev
            }
        }

        return maxPath;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<vector<int>> matrix = {
        {9, 9, 4},
        {6, 6, 8},
        {2, 1, 1}
    };
    cout << sol.longestIncreasingPath(matrix) << endl; // Output: 4
    return 0;
}

// g++ -std=c++17 Leetcode_0329_Longest_Increasing_Path_in_a_Matrix.cpp -o test      