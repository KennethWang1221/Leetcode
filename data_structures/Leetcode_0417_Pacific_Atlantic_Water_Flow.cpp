#include <iostream>
#include <vector>
#include <set>

using namespace std;

class Solution {
public:
    // DFS helper function to explore reachable cells
    void dfs(int r, int c, set<pair<int, int>>& visit, vector<vector<int>>& heights, int prevHeight) {
        int ROWS = heights.size();
        int COLS = heights[0].size();
        
        // Check if out of bounds or if the current height is lower than previous
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS || heights[r][c] < prevHeight || visit.count({r, c})) {
            return;
        }
        
        visit.insert({r, c});
        
        // Explore four directions
        dfs(r + 1, c, visit, heights, heights[r][c]);
        dfs(r - 1, c, visit, heights, heights[r][c]);
        dfs(r, c + 1, visit, heights, heights[r][c]);
        dfs(r, c - 1, visit, heights, heights[r][c]);
    }

    vector<vector<int>> pacificAtlantic_DFS(vector<vector<int>>& heights) {
        int ROWS = heights.size();
        int COLS = heights[0].size();
        
        set<pair<int, int>> pac, atl;

        // Perform DFS for the Pacific Ocean (top and left borders)
        for (int c = 0; c < COLS; c++) {
            dfs(0, c, pac, heights, heights[0][c]);
            dfs(ROWS - 1, c, atl, heights, heights[ROWS - 1][c]);
        }
        for (int r = 0; r < ROWS; r++) {
            dfs(r, 0, pac, heights, heights[r][0]);
            dfs(r, COLS - 1, atl, heights, heights[r][COLS - 1]);
        }

        // Find the intersection of cells reachable from both oceans
        vector<vector<int>> res;
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (pac.count({r, c}) && atl.count({r, c})) {
                    res.push_back({r, c});
                }
            }
        }
        return res;
    }
};

class Solution2 {
public:
    // BFS helper function to explore reachable cells
    set<pair<int, int>> bfs(queue<pair<int, int>>& q, vector<vector<int>>& heights) {
        set<pair<int, int>> visited;
        int ROWS = heights.size();
        int COLS = heights[0].size();
        
        while (!q.empty()) {
            auto [r, c] = q.front();
            q.pop();
            if (visited.count({r, c})) {
                continue;
            }
            visited.insert({r, c});
            
            // Explore four directions
            for (auto& dir : vector<pair<int, int>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                int nr = r + dir.first, nc = c + dir.second;
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && heights[nr][nc] >= heights[r][c]) {
                    q.push({nr, nc});
                }
            }
        }
        return visited;
    }

    vector<vector<int>> pacificAtlantic_BFS(vector<vector<int>>& heights) {
        int ROWS = heights.size();
        int COLS = heights[0].size();
        
        queue<pair<int, int>> pacificQueue, atlanticQueue;
        
        // Initialize queues with border cells
        for (int r = 0; r < ROWS; r++) {
            pacificQueue.push({r, 0});
            atlanticQueue.push({r, COLS - 1});
        }
        for (int c = 0; c < COLS; c++) {
            pacificQueue.push({0, c});
            atlanticQueue.push({ROWS - 1, c});
        }

        // Perform BFS for both oceans
        set<pair<int, int>> pacificReachable = bfs(pacificQueue, heights);
        set<pair<int, int>> atlanticReachable = bfs(atlanticQueue, heights);

        // Find the intersection of cells reachable from both oceans
        vector<vector<int>> res;
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (pacificReachable.count({r, c}) && atlanticReachable.count({r, c})) {
                    res.push_back({r, c});
                }
            }
        }
        return res;
    }
};

int main() {
    Solution sol;
    Solution2 sol2;
    vector<vector<int>> heights = {
        {1, 2, 2, 3, 5},
        {3, 2, 3, 4, 4},
        {2, 4, 5, 3, 1},
        {6, 7, 1, 4, 5},
        {5, 1, 1, 2, 4}
    };
    
    vector<vector<int>> res = sol.pacificAtlantic_DFS(heights);
    for (auto& row : res) {
        cout << "[" << row[0] << ", " << row[1] << "] ";
    }
    cout << endl;

    vector<vector<int>> heights2 = {
        {1, 2, 2, 3, 5},
        {3, 2, 3, 4, 4},
        {2, 4, 5, 3, 1},
        {6, 7, 1, 4, 5},
        {5, 1, 1, 2, 4}
    };
    
    vector<vector<int>> res2 = sol2.pacificAtlantic_BFS(heights2);
    for (auto& row : res2) {
        cout << "[" << row[0] << ", " << row[1] << "] ";
    }
    cout << endl;


    return 0;
}

// g++ -std=c++17 Leetcode_0417_Pacific_Atlantic_Water_Flow.cpp -o test