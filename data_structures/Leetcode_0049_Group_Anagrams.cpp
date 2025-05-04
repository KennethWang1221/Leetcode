#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <array>

class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        std::unordered_map<std::string, std::vector<std::string>> res;

        for (const std::string& s : strs) {
            std::array<int, 26> count = {0};  // Array to store character counts

            for (char c : s) {
                count[c - 'a']++;  // Increment count for the character
            }

            // Convert count array to a string key for grouping anagrams
            std::string key = "";
            for (int num : count) {
                key += std::to_string(num) + "#";  // Create a unique key based on character counts
            }

            res[key].push_back(s);  // Group anagrams using the key
        }

        std::vector<std::vector<std::string>> result;
        for (auto& entry : res) {
            result.push_back(entry.second);
        }

        return result;
    }
};

int main() {
    Solution solution;
    std::vector<std::string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
    std::vector<std::vector<std::string>> res = solution.groupAnagrams(strs);

    for (const auto& group : res) {
        for (const auto& word : group) {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0049_Group_Anagrams.cpp -o test