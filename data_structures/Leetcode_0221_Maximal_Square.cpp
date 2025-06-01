#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return 0;
        }
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<vector<int>> dp(rows + 1, vector<int>(cols + 1, 0));
        int max_side = 0;
        
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                if (matrix[i-1][j-1] == '1') {
                    dp[i][j] = min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
                    if (dp[i][j] > max_side) {
                        max_side = dp[i][j];
                    }
                }
            }
        }
        
        return max_side * max_side;
    }
};

int main() {
    Solution sol;
    vector<vector<char>> matrix = {
        {'1','0','1','0','0'},
        {'1','0','1','1','1'},
        {'1','1','1','1','1'},
        {'1','0','0','1','0'}
    };
    int result = sol.maximalSquare(matrix);
    cout << "Test case 1: " << result << endl; // Expected: 4

    matrix = {
        {'0','1'},
        {'1','0'}
    };
    result = sol.maximalSquare(matrix);
    cout << "Test case 2: " << result << endl; // Expected: 1

    matrix = {
        {'0'}
    };
    result = sol.maximalSquare(matrix);
    cout << "Test case 3: " << result << endl; // Expected: 0

    matrix = {
        {'1','1'},
        {'1','1'}
    };
    result = sol.maximalSquare(matrix);
    cout << "Test case 4: " << result << endl; // Expected: 4

    return 0;
}

// g++ -std=c++17 Leetcode_0221_Maximal_Square.cpp -o test