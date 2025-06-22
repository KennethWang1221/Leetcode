#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int numIslands_DFS(vector<vector<char>>& grid) {
        if (grid.empty() || grid[0].empty()) return 0;

        int islands = 0;
        int rows = grid.size();
        int cols = grid[0].size();
        unordered_set<string> visit;

        // Depth-first search (DFS) function
        function<void(int, int)> dfs = [&](int r, int c) {
            // Boundary/validity checks
            if (r < 0 || r >= rows || c < 0 || c >= cols || 
                grid[r][c] == '0' || 
                visit.count(to_string(r) + "," + to_string(c))) {
                return;
            }

            visit.insert(to_string(r) + "," + to_string(c));  // Mark visited
            
            // Define directions ONCE (move outside recursive calls for efficiency)
            static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            
            for (const auto& dir : directions) {
                int nr = r + dir[0];
                int nc = c + dir[1];
                dfs(nr, nc);  // Recursive call
            }
            // Explore the four possible directions
            // dfs(r + 1, c);  // Down
            // dfs(r - 1, c);  // Up
            // dfs(r, c + 1);  // Right
            // dfs(r, c - 1);  // Left
        };

        // Loop over the grid to start DFS from each land cell ('1')
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == '1' && !visit.count(to_string(r) + "," + to_string(c))) {
                    islands++;  // Found a new island
                    dfs(r, c);  // Explore this island
                }
            }
        }

        return islands;
    }



    int numIslands_BFS(vector<vector<char>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int res = 0;
        unordered_set<string> visited; // To keep track of visited cells

        // Directions for moving up, down, left, and right
        vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // Helper function for DFS
        function<void(int, int)> dfs = [&](int row, int col) {
            visited.insert(to_string(row) + "," + to_string(col));
            queue<pair<int, int>> q; 
            q.push({row, col});
            
            while (!q.empty()) {
                auto [r, c] = q.front();
                q.pop();
                
                for (const auto& dir : directions) {
                    int nr = r + dir[0], nc = c + dir[1];
                    
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && visited.find(to_string(nr) + "," + to_string(nc)) == visited.end() && grid[nr][nc] == '1') {
                        q.push({nr, nc});
                        visited.insert(to_string(nr) + "," + to_string(nc));
                    }
                }
            }
        };

        // Loop through each cell in the grid
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                // If the cell is land ('1') and not visited, start a DFS
                if (grid[row][col] == '1' && visited.find(to_string(row) + "," + to_string(col)) == visited.end()) {
                    dfs(row, col); // Start DFS to visit all connected land
                    ++res;  // Increment the island count
                }
            }
        }

        return res;
    }
};

int main() {
    Solution solution;

    // Test case: 2D grid where '1' represents land and '0' represents water
    vector<vector<char>> grid = {
        {'1','1','1','1','0'},
        {'1','1','0','1','0'},
        {'1','1','0','0','0'},
        {'0','0','0','0','0'}
    };

    int result = solution.numIslands_DFS(grid);
    cout << "Number of islands DFS: " << result << endl;  // Expected output: 1

    int result2 = solution.numIslands_BFS(grid);
    cout << "Number of islands BFS: " << result2 << endl;  // Expected output: 1

    return 0;
}

// g++ -std=c++17 Leetcode_0200_Number_of_Islands.cpp -o test