#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>

using namespace std;

class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int rows = grid.size(), cols = grid[0].size();
        
        // Min-Heap for Dijkstra-like approach
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> minH;
        minH.push({grid[0][0], 0, 0});  // {time, row, col}
        
        // Directions for the 4 possible movements (up, down, left, right)
        vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        // Set to track visited cells
        set<pair<int, int>> visited;
        visited.insert({0, 0});
        
        while (!minH.empty()) {
            auto top = minH.top();
            minH.pop();
            
            int t = top[0], r = top[1], c = top[2];

            // If we reach the bottom-right corner, return the time
            if (r == rows - 1 && c == cols - 1) {
                return t;
            }

            // Explore all 4 neighbors
            for (auto& dir : directions) {
                int nr = r + dir[0], nc = c + dir[1];
                
                // Check bounds and if the cell has been visited
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || visited.find({nr, nc}) != visited.end()) {
                    continue;
                }

                visited.insert({nr, nc});
                minH.push({max(t, grid[nr][nc]), nr, nc});  // Push the new cell with the max of current time and cell value
            }
        }

        return -1;  // If there's no path (though this case should not happen for a valid grid)
    }
};

int main() {
    Solution sol;
    vector<vector<int>> grid = {{0, 2}, {1, 3}};
    
    int res = sol.swimInWater(grid);
    cout << "Minimum time to swim in water: " << res << endl;  // Expected Output: 3
    
    return 0;
}


// g++ -std=c++17 Leetcode_0778_Swim_in_Rising_Water.cpp -o test 