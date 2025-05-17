#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        unordered_map<int, int> dp;
        dp[n] = 1;  // Base case: There's one way to decode an empty string

        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == '0') {
                dp[i] = 0;  // If the character is '0', there's no valid decoding
            } else {
                dp[i] = dp[i + 1];  // Single digit decoding
                
                // Check if the next two digits form a valid number between 10 and 26
                if (i + 1 < n && (s[i] == '1' || (s[i] == '2' && s[i + 1] >= '0' && s[i + 1] <= '6'))) {
                    dp[i] += dp[i + 2];  // Two digits decoding
                }
            }
        }

        return dp[0];  // Return the number of ways to decode the entire string
    }
};

int main() {
    Solution sol;
    string s = "12";
    int result = sol.numDecodings(s);
    cout << result << endl;  // Expected output: 2
    
    return 0;
}


// g++ -std=c++17 Leetcode_0091_Decode_Ways.cpp -o test 