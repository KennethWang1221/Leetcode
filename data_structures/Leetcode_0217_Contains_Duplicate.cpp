#include <iostream>
#include <unordered_map>
#include <vector>

class Solution {
public:
    bool containsDuplicate(std::vector<int>& nums) {
        std::unordered_map<int, int> hashmap;

        for (int n : nums) {
            if (hashmap.find(n) != hashmap.end()) {
                return true; // Duplicate found
            } else {
                hashmap[n] = 0; // Insert element into hashmap with a dummy value
            }
        }
        
        return false; // No duplicates found
    }
};

int main() {
    Solution solution;
    std::vector<int> nums = {1, 2, 3, 1};
    bool res = solution.containsDuplicate(nums);
    std::cout << res << std::endl; // Output will be 1 (true)
    return 0;
}

// g++ -std=c++17 Leetcode_0217_Contains_Duplicate.cpp -o test