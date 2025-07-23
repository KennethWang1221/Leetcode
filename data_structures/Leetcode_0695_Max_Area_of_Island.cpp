#include <iostream>
#include <vector>
#include <queue>
#include <functional>
using namespace std;

class Solution {
public:
    int maxAreaOfIsland_DFS(vector<vector<int>>& grid){
        int rows = grid.size();
        int cols = grid[0].size();
        int res = 0;
        vector<vector<bool>> visit(
            rows, // number of rows
            vector<bool>(cols,false) // each row is a vector of 'cols' booleans, all initialized to false
        );

        for (int row = 0; row < rows; row++){
            for (int col = 0; col < cols; col++){
                if ((grid[row][col] == 1 && !visit[row][col] )){
                    res = max(res, dfs(grid, visit, row, col));
                }
            }
        }

        return res;
    }

    int dfs(vector<vector<int>>& grid, vector<vector<bool>>& visit, int row, int col){
        static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        int islands = 1;
        int rows = grid.size();
        int cols = grid[0].size();
        if (
            (row < 0 || row > rows - 1 || col < 0 || col > cols-1 || visit[row][col] || grid[row][col] == 0)
        ){
            return 0;
        }

        visit[row][col] = true;
        // for (const vector<int>& dir : directions) {
        for (const auto& dir : directions){
            int r = row + dir[0];
            int c = col + dir[1];
            islands += dfs(grid, visit, r,c);
        }
        return islands;

    }

    int maxAreaOfIsland_BFS(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int res = 0;
        vector<vector<bool>> visit (rows, vector<bool>(cols, false));
        for (int row = 0; row < rows; row++){
            for (int col = 0; col < cols; col++){
                if (
                    !visit[row][col] && grid[row][col] == 1
                ){
                    res = max(res, bfs(grid, visit, row, col));
                }
            }
        }
        
        return res;
    }


    int bfs(vector<vector<int>>& grid, vector<vector<bool>>& visit, int row, int col){

        int rows = grid.size();
        int cols = grid[0].size();
        int islands = 1;
        static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        queue<pair<int, int>> q;
        q.push({row,col});
        visit[row][col] = true;
        
        while (!q.empty()){
            pair<int, int> cur = q.front();
            q.pop();
            int cur_row = cur.first;
            int cur_col = cur.second;
            for (const auto& dir:directions){
                int r = cur_row + dir[0];
                int c = cur_col + dir[1];
                if (
                    (r >=0 && r <= rows-1 && c >=0 && c <= cols-1 && !visit[r][c] && grid[r][c] == 1)
                ){
                    islands+=1;
                    q.push({r,c});
                    visit[r][c] = true;
                }
            }
        }

        return islands;
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