#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int s1n = s1.length();
        int s2n = s2.length();
        
        if (s1n > s2n) {
            return false;
        }

        vector<int> s1Count(26, 0), s2Count(26, 0);

        // Count characters in s1 and the first window of s2
        for (int i = 0; i < s1n; i++) {
            s1Count[s1[i] - 'a']++;
            s2Count[s2[i] - 'a']++;
        }

        int matches = 0;
        for (int i = 0; i < 26; i++) {
            if (s1Count[i] == s2Count[i]) {
                matches++;
            }
        }

        int l = 0;
        for (int r = s1n; r < s2n; r++) {
            if (matches == 26) {
                return true;
            }

            // Update the window with the new character at r
            int index = s2[r] - 'a';
            s2Count[index]++;
            if (s1Count[index] == s2Count[index]) {
                matches++;
            } else if (s1Count[index] + 1 == s2Count[index]) {
                matches--;
            }

            // Remove the character at l
            index = s2[l] - 'a';
            s2Count[index]--;
            if (s1Count[index] == s2Count[index]) {
                matches++;
            } else if (s1Count[index] - 1 == s2Count[index]) {
                matches--;
            }

            l++;
        }

        return matches == 26;
    }
};

int main() {
    Solution solution;
    string s1 = "abc";
    string s2 = "lecabee";

    bool result = solution.checkInclusion(s1, s2);
    cout << (result ? "True" : "False") << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0567_Permutation_in_String.cpp -o test