#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
public:
    int snakesAndLadders(vector<vector<int>>& board) {
        int n = board.size();
        reverse(board.begin(), board.end());
        
        queue<pair<int, int>> q;
        q.push({1, 0});
        unordered_set<int> visited;
        visited.insert(1);
        
        auto intToPos = [n](int square) -> pair<int, int> {
            int r = (square - 1) / n;
            int c = (square - 1) % n;
            if (r % 2 == 1) {
                c = n - 1 - c;
            }
            return make_pair(r, c);
        };
        
        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            int square = current.first;
            int moves = current.second;
            
            for (int i = 1; i <= 6; i++) {
                int nextSquare = square + i;
                auto pos = intToPos(nextSquare);
                int r = pos.first;
                int c = pos.second;
                
                if (board[r][c] != -1) {
                    nextSquare = board[r][c];
                }
                if (nextSquare == n * n) {
                    return moves + 1;
                }
                if (visited.find(nextSquare) == visited.end()) {
                    visited.insert(nextSquare);
                    q.push({nextSquare, moves + 1});
                }
            }
        }
        return -1;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> board = {
        {-1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1},
        {-1, 35, -1, -1, 13, -1},
        {-1, -1, -1, -1, -1, -1},
        {-1, 15, -1, -1, -1, -1}
    };
    int result = solution.snakesAndLadders(board);
    cout << "Result: " << result << endl; // Expected output: 4
    return 0;
}
// g++ -std=c++17 Leetcode_0909_Snakes_and_Ladders.cpp -o test