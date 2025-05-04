#include <iostream>
#include <unordered_map>
#include <vector>

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::unordered_map<int, int> premap;
        int n = nums.size();
        
        for (int i = 0; i < n; ++i) {
            if (premap.find(nums[i]) != premap.end()) {
                return {premap[nums[i]], i}; // Found the solution
            } else {
                int diff = target - nums[i];
                premap[diff] = i; // Store the difference and its index
            }
        }
        
        return {}; // Return empty if no solution found
    }
};

int main() {
    Solution solution;
    std::vector<int> nums = {2, 7, 11, 15};
    int target = 9;
    std::vector<int> res = solution.twoSum(nums, target);
    
    // Output the result
    for (int idx : res) {
        std::cout << idx << " ";
    }
    std::cout << std::endl; // Expected output: 0 1 (indexes of 2 and 7)
    
    return 0;
}
// g++ -std=c++17 Leetcode_0001_two_sum.cpp -o test