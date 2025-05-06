#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    bool validPalindromeUtil(string &s, int i, int j) {
        while (i < j) {
            if (s[i] == s[j]) {
                i++;
                j--;
            } else {
                return false;
            }
        }
        return true;
    }

    bool validPalindrome(string s) {
        int i = 0, j = s.length() - 1;

        while (i < j) {
            if (s[i] == s[j]) {
                i++;
                j--;
            } else {
                return validPalindromeUtil(s, i + 1, j) || validPalindromeUtil(s, i, j - 1);
            }
        }
        return true;
    }
};

int main() {
    Solution solution;
    string s = "cbbcc"; // You can test with other strings as well
    bool res = solution.validPalindrome(s);
    cout << (res ? "True" : "False") << endl; // Output the result
    return 0;
}
// g++ -std=c++17 Leetcode_0680_Valid_Palindrome_II.cpp -o test