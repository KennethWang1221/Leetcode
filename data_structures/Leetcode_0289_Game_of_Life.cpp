#include <vector>
using namespace std;

class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int m = board.size();
        int n = board[0].size();

        // All possible directions
        vector<pair<int, int>> dirs = {
            {-1, -1}, {-1, 0}, {-1, 1},
            {0, -1},           {0, 1},
            {1, -1},  {1, 0},   {1, 1}
        };

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int live_neighbors = 0;
                
                // Count live neighbors
                for (auto [dx, dy] : dirs) {
                    int ni = i + dx;
                    int nj = j + dy;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                        if (board[ni][nj] == 1 || board[ni][nj] == 2)
                            live_neighbors++;
                    }
                }

                // Apply rules and encode new state
                if (board[i][j] == 1) {
                    if (live_neighbors < 2 || live_neighbors > 3)
                        board[i][j] = 2; // 2 = was alive, now dead
                } else {
                    if (live_neighbors == 3)
                        board[i][j] = 3; // 3 = was dead, now alive
                }
            }
        }

        // Convert encoded states back to final values
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 2)
                    board[i][j] = 0;
                else if (board[i][j] == 3)
                    board[i][j] = 1;
            }
        }
    }
};

#include <iostream>

int main() {
    Solution sol;
    vector<vector<int>> board = {
        {0,1,0},
        {0,0,1},
        {1,1,1},
        {0,0,0}
    };

    sol.gameOfLife(board);

    cout << "[";
    for (auto& row : board) {
        cout << "[";
        for (int val : row) {
            cout << val << " ";
        }
        cout << "] ";
    }
    cout << "]" << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0289_Game_of_Life.cpp -o test