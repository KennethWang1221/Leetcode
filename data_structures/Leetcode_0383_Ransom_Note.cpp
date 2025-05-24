#include <iostream>
#include <unordered_map>
#include <string>

using namespace std;

class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> freq;

        // Count frequency of each character in magazine
        for (char c : magazine) {
            freq[c]++;
        }

        // Try to build ransomNote
        for (char c : ransomNote) {
            if (freq.find(c) == freq.end() || freq[c] == 0) {
                return false;
            }
            freq[c]--;
        }

        return true;
    }
};

// Test Case
int main() {
    Solution sol;
    cout << boolalpha << sol.canConstruct("aa", "ab") << endl; // Output: false
    cout << boolalpha << sol.canConstruct("a", "a") << endl;   // Output: true
    cout << boolalpha << sol.canConstruct("aa", "aab") << endl; // Output: true

    return 0;
}

// g++ -std=c++17 Leetcode_0383_Ransom_Note.cpp -o test