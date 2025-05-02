#include <vector>
#include <algorithm> // For std::max
#include <iostream>
using namespace std;

class Solution {

private:
    int dfs(vector<vector<int>>& grid, int row, int col, vector<vector<bool>>& visited) {
        if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size() 
            || grid[row][col] != 1 || visited[row][col]) {
            return 0;
        }
        visited[row][col] = true;
        int area = 1;
        // Directions: up, down, left, right
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (auto& dir : directions) {
            area += dfs(grid, row + dir.first, col + dir.second, visited);
        }
        return area;
    }

public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int rows = grid.size();
        if (rows == 0) return 0;
        int cols = grid[0].size();
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));
        int max_area = 0;
        
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (grid[row][col] == 1 && !visited[row][col]) {
                    max_area = max(max_area, dfs(grid, row, col, visited));
                }
            }
        }
        return max_area;
    }
    
};

int main() {
    Solution sol;
    vector<vector<int>> grid = {
        {0,0,1,0,0,0,0,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,1,1,0,0,0},
        {0,1,1,0,1,0,0,0,0,0,0,0,0},
        {0,1,0,0,1,1,0,0,1,0,1,0,0},
        {0,1,0,0,1,1,0,0,1,1,1,0,0},
        {0,0,0,0,0,0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0,0,1,1,1,0,0,0},
        {0,0,0,0,0,0,0,1,1,0,0,0,0}
    };
    int res = sol.maxAreaOfIsland(grid);
    cout << "Result: " << res << endl; // Expected output: 6
    return 0;
}
// g++ -std=c++17 Leetcode_0695_Max_Area_of_Island.cpp -o test