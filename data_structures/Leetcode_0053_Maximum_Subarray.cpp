#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        // Initialize with the first element
        int current_sum = nums[0];
        int max_sum = nums[0];
        
        // Iterate through the array starting from the second element
        for (int i = 1; i < nums.size(); ++i) {
            // Greedily decide whether to add the current number to the existing subarray or start a new one
            current_sum = max(nums[i], current_sum + nums[i]);
            
            // Update the max_sum if the current_sum is greater
            max_sum = max(max_sum, current_sum);
        }
        
        return max_sum;
    }
};

int main() {
    Solution solution;
    
    // Example usage
    vector<int> nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "Maximum subarray sum: " << solution.maxSubArray(nums) << endl;  // Expected output: 6
    
    return 0;
}

// g++ -std=c++17 Leetcode_0053_Maximum_Subarray.cpp -o test 