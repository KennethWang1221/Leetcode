#include <iostream>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int totalNQueens(int n) {
        int answer = 0;
        unordered_set<int> cols, posdiag, negdiag;
        backtrack(n, 0, answer, cols, posdiag, negdiag);
        return answer;
    }

private:
    void backtrack(int n, int i, int& answer, unordered_set<int>& cols, unordered_set<int>& posdiag, unordered_set<int>& negdiag) {
        if (i == n) {
            answer++;  // If all queens are placed successfully, increment the answer
            return;
        }

        for (int j = 0; j < n; ++j) {
            // Check if the current position is safe
            if (cols.count(j) || posdiag.count(i + j) || negdiag.count(i - j)) {
                continue;
            }

            // Mark the current position as occupied
            cols.insert(j);
            posdiag.insert(i + j);
            negdiag.insert(i - j);

            // Recurse to place the next queen in the next row
            backtrack(n, i + 1, answer, cols, posdiag, negdiag);

            // Backtrack by removing the current queen
            cols.erase(j);
            posdiag.erase(i + j);
            negdiag.erase(i - j);
        }
    }
};

int main() {
    Solution solution;
    int n = 4;

    int result = solution.totalNQueens(n);
    cout << result << endl;  // Expected output: 2

    return 0;
}


// g++ -std=c++17 Leetcode_0052_N_Queens_II.cpp -o test