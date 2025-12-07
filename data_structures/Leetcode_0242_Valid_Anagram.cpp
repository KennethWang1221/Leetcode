#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class Solution {
public:
    bool isAnagram(std::string s, std::string t) {
        int s_n = s.length();
        int t_n = t.length();

        if (s_n != t_n) {
            return false; // If lengths are not equal, they cannot be anagrams
        }

        std::unordered_map<char, int> s_dict;
        std::unordered_map<char, int> t_dict;

        // Populate s_dict with character frequencies from string s
        for (char c : s) {
            s_dict[c]++;
        }

        // Populate t_dict with character frequencies from string t
        for (char c : t) {
            t_dict[c]++;
        }

        // Compare frequency maps of both strings
        for (const auto& entry : s_dict) {
            char n = entry.first;
            int c = entry.second;
            if (t_dict.find(n) == t_dict.end()) {
                return false; // Character not found in t_dict
            }
            if (c != t_dict[n]) {
                return false; // Frequency mismatch
            }
        }

        return true; // All checks passed, the strings are anagrams
    }


    bool isAnagram_method2(string s, string t) {

        int s_n = s.size();
        int t_n = t.size();
        if (s_n != t_n){
            return false;
        }

        unordered_map<char, int> hashmap_s;
        unordered_map<char, int> hashmap_t;

        for (int i = 0; i < s_n; i++){
            hashmap_s[s[i]]++;
            hashmap_t[t[i]]++;
        }

        if (hashmap_s != hashmap_t){
            return false;
        }else{
            return true;
        }
    }
};

int main() {
    Solution solution;
    std::string s = "anagram";
    std::string t = "nagaram";
    bool res = solution.isAnagram(s, t);
    std::cout << res << std::endl; // Output will be 1 (true)

    bool res2 = solution.isAnagram_method2(s, t);
    cout << res2 << std::endl; // Output will be 1 (true)
    return 0;
}

// g++ -std=c++17 Leetcode_0242_Valid_Anagram.cpp -o test