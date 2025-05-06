#include <iostream>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    int characterReplacement(string s, int k) {
        unordered_map<char, int> count;  // Hashmap to count character frequencies
        int l = 0, maxf = 0, res = 0;
        int n = s.size();

        for (int r = 0; r < n; r++) {
            count[s[r]]++;  // Increment count of the current character
            maxf = max(maxf, count[s[r]]);  // Update max frequency of any character

            // If the window size minus the frequency of the most frequent character exceeds k, shrink the window
            while ((r - l + 1) - maxf > k) {
                count[s[l]]--;  // Decrement the count of the character at the left of the window
                l++;  // Shrink the window from the left
            }

            // Update the result with the size of the valid window
            res = max(res, r - l + 1);
        }

        return res;
    }
};

int main() {
    Solution solution;
    string s = "AABABBA";
    int k = 1;
    
    int result = solution.characterReplacement(s, k);
    cout << "Longest substring length: " << result << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0424_Longest_Repeating_Character_Replacement.cpp -o test