#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int rows = obstacleGrid.size();
        int cols = obstacleGrid[0].size();
        vector<vector<long long>> dp (rows+1, vector<long long>(cols+1,0));
        dp[rows][cols-1] = 1;

        for (int row = rows-1; row > -1; row --){
            for (int col = cols-1; col > -1; col --){
                if (obstacleGrid[row][col] == 1){
                    dp[row][col] = 0;
                } else {
                    dp[row][col] = dp[row+1][col] + dp[row][col+1];
                }
            }
        }

        return dp[0][0];

    }
};

int main() {
    Solution sol;
    vector<vector<int>> obstacleGrid = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    cout << sol.uniquePathsWithObstacles(obstacleGrid) << endl; // Expected output: 2
    return 0;
}

// g++ -std=c++17 Leetcode_0063_Unique_Paths_II.cpp -o test