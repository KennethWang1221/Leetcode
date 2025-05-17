#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
    int countSubstrings(string s) {
        int res = 0;
        int n = s.size();

        // Check for palindromes with center at each character and between every two characters
        for (int i = 0; i < n; ++i) {
            // Odd-length palindromes centered at s[i]
            int l = i, r = i;
            while (l >= 0 && r < n && s[l] == s[r]) {
                res++;
                l--;
                r++;
            }

            // Even-length palindromes centered between s[i] and s[i+1]
            l = i;
            r = l + 1;
            while (l >= 0 && r < n && s[l] == s[r]) {
                res++;
                l--;
                r++;
            }
        }

        return res;
    }
};

int main() {
    Solution sol;
    string s = "aaa";
    int result = sol.countSubstrings(s);
    cout << result << endl;  // Expected output: 6 (palindromic substrings: "a", "a", "a", "aa", "aa", "aaa")
    
    return 0;
}


// g++ -std=c++17 Leetcode_0647_Palindromic_Substrings.cpp -o test