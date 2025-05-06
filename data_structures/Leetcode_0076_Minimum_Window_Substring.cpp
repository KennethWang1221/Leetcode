#include <iostream>
#include <unordered_map>
#include <climits>
using namespace std;

class Solution {
public:
    string minWindow(string s, string t) {
        if (t.empty()) {
            return "";
        }

        unordered_map<char, int> countT, window;
        for (char c : t) {
            countT[c]++;
        }

        int have = 0, need = countT.size();
        int resLen = INT_MAX;
        int l = 0;
        int resL = -1, resR = -1;

        for (int r = 0; r < s.size(); r++) {
            char c = s[r];
            window[c]++;

            // If this character satisfies the requirement for the window
            if (countT.find(c) != countT.end() && window[c] == countT[c]) {
                have++;
            }

            // When we have a valid window, try to shrink it from the left
            while (have == need) {
                // Update the result if the current window is smaller
                if (r - l + 1 < resLen) {
                    resL = l;
                    resR = r;
                    resLen = r - l + 1;
                }

                // Try to shrink the window by moving the left pointer
                window[s[l]]--;
                if (countT.find(s[l]) != countT.end() && window[s[l]] < countT[s[l]]) {
                    have--;
                }
                l++;
            }
        }

        return resL == -1 ? "" : s.substr(resL, resR - resL + 1);
    }
};

int main() {
    Solution solution;
    string s = "ADOBECODEBANC";
    string t = "ABC";

    string result = solution.minWindow(s, t);
    cout << "Minimum window substring: " << result << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0076_Minimum_Window_Substring.cpp -o test