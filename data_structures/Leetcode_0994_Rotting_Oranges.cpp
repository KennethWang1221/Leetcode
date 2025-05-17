#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    // BFS Approach
    int orangesRotting_BFS(vector<vector<int>>& grid) {
        queue<pair<int, int>> q;
        int fresh = 0;
        int time = 0;
        int rows = grid.size();
        int cols = grid[0].size();

        // Count fresh oranges and add initial rotten ones to the queue
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 1) {
                    fresh++;
                } else if (grid[r][c] == 2) {
                    q.push({r, c});
                }
            }
        }

        vector<vector<int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while (fresh > 0 && !q.empty()) {
            int length = q.size();
            for (int i = 0; i < length; i++) {
                auto [row, col] = q.front();
                q.pop();
                for (auto& dir : directions) {
                    int r = row + dir[0];
                    int c = col + dir[1];
                    if (r >= 0 && r < rows && c >= 0 && c < cols && grid[r][c] == 1) {
                        grid[r][c] = 2;
                        q.push({r, c});
                        fresh--;
                    }
                }
            }
            time++;
        }

        return (fresh == 0) ? time : -1;
    }

    // DFS Approach
    void dfs(int r, int c, int time, vector<vector<int>>& grid, int rows, int cols, int& fresh) {
        vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (auto& dir : directions) {
            int nr = r + dir.first;
            int nc = c + dir.second;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                if (grid[nr][nc] == 1) {
                    grid[nr][nc] = time + 3;  // Mark with a timestamp (offset by 2)
                    fresh--;
                    dfs(nr, nc, time + 1, grid, rows, cols, fresh);
                } else if (grid[nr][nc] > 2 && (grid[nr][nc] - 2) > time + 1) {
                    grid[nr][nc] = time + 3;
                    dfs(nr, nc, time + 1, grid, rows, cols, fresh);
                }
            }
        }
    }

    int orangesRotting(vector<vector<int>>& grid, bool useBFS = true) {
        if (useBFS) {
            return orangesRotting_BFS(grid);
        }

        int rows = grid.size();
        int cols = grid[0].size();
        int fresh = 0;
        vector<pair<int, int>> rotten;

        // Count fresh oranges and record initial rotten ones
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 1) {
                    fresh++;
                } else if (grid[r][c] == 2) {
                    rotten.push_back({r, c});
                }
            }
        }

        // Start DFS from each initially rotten orange
        for (auto& [r, c] : rotten) {
            dfs(r, c, 0, grid, rows, cols, fresh);
        }

        int max_time = 0;
        // Find the maximum time taken to rot an orange
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] > 2) {
                    max_time = max(max_time, grid[r][c] - 2);
                }
            }
        }

        // If there are still fresh oranges, return -1. Otherwise, return the time.
        return (fresh == 0) ? max_time : -1;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> grid = {{2, 1, 1}, {1, 1, 0}, {0, 1, 1}};
    
    // Use BFS
    int resBFS = sol.orangesRotting(grid, true);
    cout << "BFS Result: " << resBFS << endl;
    vector<vector<int>> grid2 = {{2, 1, 1}, {1, 1, 0}, {0, 1, 1}};
    // Use DFS
    int resDFS = sol.orangesRotting(grid2, false);
    cout << "DFS Result: " << resDFS << endl;

    return 0;
}



// g++ -std=c++17 Leetcode_0994_Rotting_Oranges.cpp -o test