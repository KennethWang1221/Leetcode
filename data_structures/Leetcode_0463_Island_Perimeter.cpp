#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        unordered_set<string> visit;
        int rows = grid.size();
        int cols = grid[0].size();
        int perim = 0;

        // Helper function for Depth-First Search
        function<int(int, int)> dfs = [&](int r, int c) {
            // Out of bounds or water (0)
            if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] == 0) {
                return 1;
            }
            string pos = to_string(r) + "," + to_string(c);
            if (visit.count(pos)) {
                return 0;
            }

            visit.insert(pos);

            int perimeter = 0;
            perimeter += dfs(r - 1, c); // Up
            perimeter += dfs(r + 1, c); // Down
            perimeter += dfs(r, c - 1); // Left
            perimeter += dfs(r, c + 1); // Right

            return perimeter;
        };

        // Iterate through the grid to start DFS from every land cell (1)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == 1) {
                    perim += dfs(r, c);
                }
            }
        }

        return perim;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> grid = {
        {0, 1, 0, 0},
        {1, 1, 1, 0},
        {0, 1, 0, 0},
        {1, 1, 0, 0}
    };

    int result = solution.islandPerimeter(grid);
    cout << "Island Perimeter: " << result << endl;  // Expected output: 16

    return 0;
}


// g++ -std=c++17 Leetcode_0463_Island_Perimeter.cpp -o test