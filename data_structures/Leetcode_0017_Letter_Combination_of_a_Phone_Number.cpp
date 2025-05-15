#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> result;
        string s = "";
        int n = digits.size();
        
        // If the input digits are empty, return an empty result
        if (n == 0) return result;

        backtracking(digits, 0, s, result);
        return result;
    }

private:
    // Mapping from digits to letters
    vector<string> letterMap = {
        "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
    };

    void backtracking(const string& digits, int i, string& s, vector<string>& result) {
        if (i == digits.size()) {
            result.push_back(s);  // If the combination is complete, add to result
            return;
        }

        // Get the current digit and corresponding letters
        int digit = digits[i] - '0';
        for (char c : letterMap[digit]) {
            s += c;  // Add the current character to the combination
            backtracking(digits, i + 1, s, result);  // Recurse with the next digit
            s.pop_back();  // Backtrack, remove the last added character
        }
    }
};

int main() {
    Solution solution;
    string digits = "23";

    vector<string> result = solution.letterCombinations(digits);

    // Print the result
    for (const string& combination : result) {
        cout << combination << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0017_Letter_Combination_of_a_Phone_Number.cpp -o test