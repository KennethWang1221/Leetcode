#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> comb;
        backtracking(s, 0, comb, res);
        return res;
    }

private:
    // Function to check if a substring is a palindrome
    bool isPali(const string& s, int l, int r) {
        while (l < r) {
            if (s[l] != s[r]) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    // Backtracking function to generate all palindromic partitions
    void backtracking(const string& s, int start, vector<string>& comb, vector<vector<string>>& res) {
        if (start >= s.length()) {
            res.push_back(comb);  // If all characters are used, add the current combination to the result
            return;
        }

        for (int end = start; end < s.length(); ++end) {
            if (isPali(s, start, end)) {
                comb.push_back(s.substr(start, end - start + 1));  // Add the palindrome substring
                backtracking(s, end + 1, comb, res);  // Recurse for the remaining part of the string
                comb.pop_back();  // Backtrack by removing the last added substring
            }
        }
    }
};

int main() {
    Solution solution;
    string s = "aab";

    vector<vector<string>> result = solution.partition(s);

    // Print the result
    for (const auto& partition : result) {
        cout << "[ ";
        for (const auto& str : partition) {
            cout << str << " ";
        }
        cout << "]" << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0131_Palindrome_Partitioning.cpp -o test