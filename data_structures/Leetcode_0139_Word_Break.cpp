#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> dp(n + 1, false);  // dp[i] will be true if s[0...i-1] can be segmented
        dp[0] = true;  // Base case: an empty string can always be segmented

        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());  // Convert wordDict to a set for faster lookup

        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (dp[j] && wordSet.count(s.substr(j, i - j))) {  // If substring s[j...i-1] is in wordDict
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];  // dp[n] tells if the whole string can be segmented
    }
};

int main() {
    Solution sol;
    string s = "leetcode";
    vector<string> wordDict = {"leet", "code"};
    
    bool result = sol.wordBreak(s, wordDict);
    cout << (result ? "True" : "False") << endl;  // Expected output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0139_Word_Break.cpp -o test