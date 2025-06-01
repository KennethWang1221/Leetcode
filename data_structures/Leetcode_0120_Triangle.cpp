#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        if (n == 0) return 0;
        
        vector<int> dp = triangle.back();
        
        for (int row = n - 2; row >= 0; --row) {
            for (int col = 0; col <= row; ++col) {
                dp[col] = triangle[row][col] + min(dp[col], dp[col + 1]);
            }
        }
        
        return dp[0];
    }
};

int main() {
    Solution sol;
    vector<vector<int>> triangle = {{2}, {3, 4}, {6, 5, 7}, {4, 1, 8, 3}};
    int result = sol.minimumTotal(triangle);
    cout << "Test case 1: " << result << endl; // Expected: 11

    triangle = {{-10}};
    result = sol.minimumTotal(triangle);
    cout << "Test case 2: " << result << endl; // Expected: -10

    triangle = {{1}, {2, 3}, {4, 5, 6}};
    result = sol.minimumTotal(triangle);
    cout << "Test case 3: " << result << endl; // Expected: 7

    return 0;
}

// g++ -std=c++17 Leetcode_0120_Triangle.cpp -o test
