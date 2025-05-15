#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
using namespace std;

class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int ROWS = board.size();
        int COLS = board[0].size();
        unordered_set<string> path;

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (dfs(board, word, r, c, 0, path)) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    bool dfs(vector<vector<char>>& board, string& word, int r, int c, int i, unordered_set<string>& path) {
        if (i == word.size()) {
            return true;
        }
        if (r < 0 || c < 0 || r >= board.size() || c >= board[0].size() ||
            word[i] != board[r][c] || path.count(to_string(r) + "," + to_string(c))) {
            return false;
        }

        path.insert(to_string(r) + "," + to_string(c));  // mark current position as visited

        bool res = dfs(board, word, r + 1, c, i + 1, path) ||
                   dfs(board, word, r - 1, c, i + 1, path) ||
                   dfs(board, word, r, c + 1, i + 1, path) ||
                   dfs(board, word, r, c - 1, i + 1, path);

        path.erase(to_string(r) + "," + to_string(c));  // backtrack, unmark the position
        return res;
    }
};

int main() {
    Solution solution;
    vector<vector<char>> board = {
        {'A', 'B', 'C', 'E'},
        {'S', 'F', 'C', 'S'},
        {'A', 'D', 'E', 'E'}
    };
    string word = "ABCCED";

    bool result = solution.exist(board, word);
    cout << (result ? "True" : "False") << endl;  // Output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0079_Word_Search.cpp -o test