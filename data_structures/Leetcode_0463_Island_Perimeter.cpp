#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution{
    public:
        int islandPerimeter(vector<vector<int>>& grid){
            int rows = grid.size();
            int cols = grid[0].size();
            vector<vector<bool>> visit (
                rows,
                vector<bool>(cols,false)
            );

            for (int row = 0; row < rows; row ++){
                for (int col = 0; col < cols; col++){
                    if (grid[row][col] == 1 && (!visit[row][col])){
                        return dfs(grid, visit, row, col);
                    }
                }
            }
            return 0;
        }

        int dfs(vector<vector<int>>& grid, vector<vector<bool>>& visit, int row, int col){
            if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size() || grid[row][col] == 0  ){
                return 1;
            }

            if (visit[row][col]){
                return 0;
            }
            
            visit[row][col] = true;
            int perimeter = 0;
            static const vector<vector<int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};

            for (const vector<int>& dir : directions){
                int r = dir[0] + row;
                int c = dir[1] + col;
                perimeter += dfs(grid, visit, r, c);
            }

            return perimeter;
        }

        int dfs2(vector<vector<int>>& grid, vector<vector<bool>>& visit, int row, int col){
            if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size() || grid[row][col] == 0){
                return 1;
            }
            if (visit[row][col]){
                return 0;
            }
    
            visit[row][col] = true;
            int perimeter = 0;
            perimeter += dfs(grid, visit, row+1, col);
            perimeter += dfs(grid, visit, row-1, col);
            perimeter += dfs(grid, visit, row, col-1);
            perimeter += dfs(grid, visit, row, col+1);
            return perimeter;
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