#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashmap;
        int n = nums.size();
        vector<int> res;

        for (int i = 0; i < n; ++i) {
            int remain = target - nums[i];
            if (hashmap.find(remain) != hashmap.end()) {
                res.push_back(hashmap[remain]);
                res.push_back(i);
                return res;
                // return {premap[nums[i]], i}; // Found the solution
            } else {
                hashmap.insert({nums[i], i});
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