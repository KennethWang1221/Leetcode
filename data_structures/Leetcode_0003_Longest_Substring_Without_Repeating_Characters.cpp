#include <iostream>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> window;
        int max_len = 0;
        int left = 0;
        int n = s.size();
        
        for (int right = 0; right < n; ++right) {
            while (window.find(s[right]) != window.end()) {
                window.erase(s[left]);
                left++;
            }
            window.insert(s[right]);
            max_len = max(max_len, right - left + 1);
        }
        return max_len;
    }
};

// Add this main function to test your code
int main() {
    Solution sol;
    cout << "Test 1 (\"abcabcbb\"): " << sol.lengthOfLongestSubstring("abcabcbb") << endl; // Expected: 3
    cout << "Test 2 (\"bbbbb\"): " << sol.lengthOfLongestSubstring("bbbbb") << endl;       // Expected: 1
    cout << "Test 3 (\"pwwkew\"): " << sol.lengthOfLongestSubstring("pwwkew") << endl;     // Expected: 3
    return 0;
}

// g++ -std=c++17 Leetcode_0003_Longest_Substring_Without_Repeating_Characters.cpp -o test