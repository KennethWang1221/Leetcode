#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    bool wordPattern(string pattern, string s) {
        // Split the string into words
        vector<string> words = split(s, ' ');
        
        // If lengths don't match, return false
        if (words.size() != pattern.size()) {
            return false;
        }

        unordered_map<char, string> ptos; // Pattern to string
        unordered_map<string, char> stop; // String to pattern
        for (int i = 0; i < pattern.size(); ++i) {
            char c = pattern[i];
            string word = words[i];

            // Check both mappings
            if ((ptos.count(c) && ptos[c] != word) ||
                (stop.count(word) && stop[word] != c)) {
                return false;
            }

            ptos[c] = word;
            stop[word] = c;
        }

        return true;
    }

private:
    // Helper function to split string based on delimiter
    vector<string> split(const string& s, char delimiter) {
        vector<string> tokens;
        string token;
        istringstream ss(s);
        while (getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
};

// Test Case
int main() {
    Solution sol;
    
    string pattern1 = "abba";
    string s1 = "dog cat cat dog";
    cout << boolalpha << sol.wordPattern(pattern1, s1) << endl; // Expected: true

    string pattern2 = "abba";
    string s2 = "dog cat cat fish";
    cout << boolalpha << sol.wordPattern(pattern2, s2) << endl; // Expected: false

    string pattern3 = "abba";
    string s3 = "dog dog dog dog";
    cout << boolalpha << sol.wordPattern(pattern3, s3) << endl; // Expected: false

    return 0;
}
// g++ -std=c++17 Leetcode_0290_Word_Pattern.cpp -o test
