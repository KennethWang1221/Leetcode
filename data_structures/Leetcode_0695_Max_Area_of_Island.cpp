#include <iostream>
#include <vector>
#include <queue>
#include <functional>
using namespace std;

class Solution {
public:
    int maxAreaOfIsland_DFS(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int max_area = 0;

        // Directions for moving up, down, left, and right
        vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // DFS helper function to calculate the area of the island
        function<int(int, int)> dfs = [&](int r, int c) {
            // Base case: check bounds and if the cell is land
            if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] == 0) {
                return 0;
            }

            grid[r][c] = 0;  // Mark the cell as visited
            int area = 1;  // Current cell counts as area

            // Explore all four directions
            for (const auto& dir : directions) {
                area += dfs(r + dir[0], c + dir[1]);
            }

            return area;
        };

        // Iterate over each cell in the grid
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == 1) {  // Found a new island
                    max_area = max(max_area, dfs(r, c));  // Update max area
                }
            }
        }

        return max_area;
    }

    int maxAreaOfIsland_BFS(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int max_area = 0;

        // Directions for moving up, down, left, and right
        vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // BFS helper function to calculate the area of the island
        auto bfs = [&](int r, int c) {
            int area = 0;
            queue<pair<int, int>> q;
            q.push({r, c});
            grid[r][c] = 0;  // Mark the cell as visited

            while (!q.empty()) {
                auto [row, col] = q.front();
                q.pop();
                area++;

                // Explore all four directions
                for (const auto& dir : directions) {
                    int nr = row + dir[0], nc = col + dir[1];

                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                        grid[nr][nc] = 0;  // Mark as visited
                        q.push({nr, nc});
                    }
                }
            }

            return area;
        };

        // Iterate over each cell in the grid
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == 1) {  // Found a new island
                    max_area = max(max_area, bfs(r, c));  // Update max area
                }
            }
        }

        return max_area;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> grid = {
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {0, 1, 0, 1, 1},
        {0, 0, 0, 1, 0}
    };

    vector<vector<int>> grid2 = {
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {0, 1, 0, 1, 1},
        {0, 0, 0, 1, 0}
    };
    // Test DFS method
    cout << "Max area of island (DFS): " << solution.maxAreaOfIsland_DFS(grid) << endl;  // Expected output: 6
    // Test BFS method
    cout << "Max area of island (BFS): " << solution.maxAreaOfIsland_BFS(grid2) << endl;  // Expected output: 6

    return 0;
}

// g++ -std=c++17 Leetcode_0695_Max_Area_of_Island.cpp -o test