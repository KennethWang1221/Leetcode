#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
    public:
        int numIslands_DFS(vector<vector<char>> &grid){
            if (grid.empty() || grid[0].empty()) {
                return 0;
            }

            int res = 0;
            int rows = grid.size();
            int cols = grid[0].size();
            vector<vector<bool>> visit(
                rows, // number of rows
                vector<bool>(cols,false) // each row is a vector of 'cols' booleans, all initialized to false
            );

            for (int row = 0; row < rows ; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if (grid[row][col] == '1' && !visit[row][col]) {
                        dfs(grid,row,col,visit);
                        res++;
                    }

                }
            }
            return res;
        }


    public:
        void dfs(vector<vector<char>>& grid, int row, int col, vector<vector<bool>>& visit) {
            int rows = grid.size();
            int cols = grid[0].size();

            if (row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == '0' || visit[row][col]) {
                return;
            }

            // Mark current cell as visited
            visit[row][col] = true;
            // Define directions ONCE (move outside recursive calls for efficiency)
            static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};

            // vector<int>& dir and vector<int> &dir   // same   
            // vector<int>& dir popular written way since A reference is part of the type. So vector<int>& is the full type: "reference to a vector of int". This style helps emphasize: "dir is a reference to a vector<int>" 
            for (const vector<int>& dir : directions) {
                int r = row + dir[0];
                int c = col + dir[1];
                dfs(grid, r, c, visit);  // Recursive call
            }

        }

    public:
        int numIslands_BFS(vector<vector<char>>& grid) {
            int rows = grid.size();
            int cols = grid[0].size();
            int res = 0;
            vector<vector<bool>> visit (rows, vector<bool> (cols, false));
            
            for (int row=0; row < rows; ++row){
                for (int col=0; col < cols; ++col){
                    if (grid[row][col] == '1' && !visit[row][col]){
                        if (bfs(grid,row,col,visit)){
                            res+=1;
                        }
                    }
                }
            }
            return res;
            
        }

        bool bfs(vector<vector<char>>& grid, int row, int col, vector<vector<bool>>& visit ){
            int rows = grid.size();
            int cols = grid[0].size();
            static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            queue<pair<int, int>> q;
            q.push({row, col});
            visit[row][col] = true;
            
            while (!q.empty()){
                pair<int, int> curr = q.front();
                q.pop();
                
                int row = curr.first;
                int col = curr.second;

                for (const vector<int>& dir: directions){
                    int r = row + dir[0];
                    int c = col + dir[1];
                    if (
                        (r >= 0 && r<rows && c>=0 && c<cols && grid[r][c] =='1' && !visit[r][c])
                    ){
                        q.push({r,c});
                        visit[r][c] = true;
                    }
                }

            }
            return true;
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