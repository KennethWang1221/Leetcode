#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>

using namespace std;

// Custom hash function for pair<int, int>
struct hash_pair {
    template <class T1, class T2>
    size_t operator ()(const pair<T1, T2>& p) const {
        auto h1 = hash<T1>{}(p.first);  // Hash the first element of the pair
        auto h2 = hash<T2>{}(p.second); // Hash the second element of the pair
        return h1 ^ (h2 << 1);  // Combine the two hash values
    }
};

class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        unordered_map<int, unordered_set<char>> cols;      // Columns check
        unordered_map<int, unordered_set<char>> rows;      // Rows check
        unordered_map<pair<int, int>, unordered_set<char>, hash_pair> squares;  // 3x3 Squares check

        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                if (board[r][c] == '.') continue;  // Skip empty cells

                // Check if the current number is already seen in the row, column, or square
                if (rows[r].count(board[r][c]) || 
                    cols[c].count(board[r][c]) || 
                    squares[{r / 3, c / 3}].count(board[r][c])) {
                    return false;  // Invalid Sudoku if number is already seen
                }

                // Add the current number to the respective row, column, and square sets
                rows[r].insert(board[r][c]);
                cols[c].insert(board[r][c]);
                squares[{r / 3, c / 3}].insert(board[r][c]);
            }
        }

        return true;  // Return true if the Sudoku is valid
    }
};

int main() {
    Solution solution;

    vector<vector<char>> board = {
        {'5','3','.','.','7','.','.','.','.'},
        {'6','.','.','1','9','5','.','.','.'},
        {'.','9','8','.','.','.','.','6','.'},
        {'8','.','.','.','6','.','.','.','3'},
        {'4','.','.','8','.','3','.','.','1'},
        {'7','.','.','.','2','.','.','.','6'},
        {'.','6','.','.','.','.','2','8','.'},
        {'.','.','.','4','1','9','.','.','5'},
        {'.','.','.','.','8','.','.','7','9'}
    };

    bool res = solution.isValidSudoku(board);
    cout << (res ? "True" : "False") << endl;  // Expected output: True

    return 0;
}

// g++ -std=c++17 Leetcode_0036_Valid_Sudoku.cpp -o test