#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> board(n, string(n, '.'));  // Initialize the board with '.' representing empty spaces
        unordered_set<int> col, posDiag, negDiag;  // Sets to track columns and diagonals
        backtrack(n, 0, board, res, col, posDiag, negDiag);
        return res;
    }

private:
    void backtrack(int n, int r, vector<string>& board, vector<vector<string>>& res,
                   unordered_set<int>& col, unordered_set<int>& posDiag, unordered_set<int>& negDiag) {
        if (r == n) {
            res.push_back(board);  // Add the current valid board configuration to the result
            return;
        }

        for (int c = 0; c < n; ++c) {
            if (col.count(c) || posDiag.count(r + c) || negDiag.count(r - c)) {
                continue;  // Skip if the position is under attack
            }

            // Place the queen
            col.insert(c);
            posDiag.insert(r + c);
            negDiag.insert(r - c);
            board[r][c] = 'Q';

            // Recurse to place queens in the next row
            backtrack(n, r + 1, board, res, col, posDiag, negDiag);

            // Backtrack: remove the queen and try the next column
            col.erase(c);
            posDiag.erase(r + c);
            negDiag.erase(r - c);
            board[r][c] = '.';  // Reset the cell
        }
    }
};

int main() {
    Solution solution;
    int n = 4;

    vector<vector<string>> result = solution.solveNQueens(n);

    // Print the result
    for (const auto& board : result) {
        for (const auto& row : board) {
            cout << row << endl;
        }
        cout << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0051_N_Queens.cpp -o test