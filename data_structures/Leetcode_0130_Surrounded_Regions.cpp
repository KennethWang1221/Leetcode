#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    void dfs(int row, int col, vector<vector<char>>& board, int rows, int cols) {
        if (row >= 0 && col >= 0 && row < rows && col < cols && board[row][col] == 'O') {
            board[row][col] = 'E';  // Mark as 'E' to avoid marking it as 'X'
            vector<vector<int>> directions = {{1, 0}, {-1, 0}, {0, -1}, {0, 1}};
            for (auto& dir : directions) {
                int dr = dir[0], dc = dir[1];
                dfs(row + dr, col + dc, board, rows, cols);  // Recursively visit all 'O's
            }
        }
    }

    void solve(vector<vector<char>>& board) {
        int rows = board.size();
        int cols = board[0].size();

        // Perform DFS on the borders to mark 'O's connected to the borders as 'E'
        for (int row = 0; row < rows; row++) {
            if (board[row][0] == 'O') {
                dfs(row, 0, board, rows, cols);
            }
            if (board[row][cols - 1] == 'O') {
                dfs(row, cols - 1, board, rows, cols);
            }
        }

        for (int col = 0; col < cols; col++) {
            if (board[0][col] == 'O') {
                dfs(0, col, board, rows, cols);
            }
            if (board[rows - 1][col] == 'O') {
                dfs(rows - 1, col, board, rows, cols);
            }
        }

        // Flip the remaining 'O's to 'X' and 'E' back to 'O'
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (board[row][col] == 'O') {
                    board[row][col] = 'X';  // Flip isolated 'O's to 'X'
                } else if (board[row][col] == 'E') {
                    board[row][col] = 'O';  // Flip 'E' back to 'O'
                }
            }
        }
    }
};

int main() {
    Solution sol;
    vector<vector<char>> board = {
        {'X', 'X', 'X', 'X'},
        {'X', 'O', 'O', 'X'},
        {'X', 'X', 'O', 'X'},
        {'X', 'O', 'X', 'X'}
    };

    sol.solve(board);

    // Output the modified board
    for (const auto& row : board) {
        for (char cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0130_Surrounded_Regions.cpp -o test