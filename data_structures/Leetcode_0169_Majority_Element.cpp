#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int majorityElement(std::vector<int>& nums) {
        int n = nums.size();
        int appears = n / 2;
        std::unordered_map<int, int> record;

        // Count the frequency of each element in the vector
        for (int i = 0; i < n; ++i) {
            record[nums[i]]++;
        }

        // Create a vector of pairs (value, frequency) and sort by frequency
        std::vector<std::pair<int, int>> res;
        for (const auto& entry : record) {
            res.push_back({entry.second, entry.first});
        }

        std::sort(res.begin(), res.end(), std::greater<std::pair<int, int>>());

        // Return the element with the highest frequency
        return res[0].second;
    }
};

int main() {
    Solution solution;
    std::vector<int> nums = {3, 2, 3};
    
    int result = solution.majorityElement(nums);
    std::cout << "Majority Element: " << result << std::endl; // Output should be 3
    
    return 0;
}
// g++ -std=c++17 Leetcode_0169_Majority_Element.cpp -o test