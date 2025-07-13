#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

class Solution {
public:
    int maxSubArray_method1(vector<int>& nums) {
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

    int maxSubArray_method2(vector<int>& nums) {
        int n = nums.size();
        float res = -INFINITY;
        int count = 0;

        for (int i = 0; i < n; i++){
            count += nums[i];
            if (count > res){
                res = count;
            }
            if (count <= 0){
                count = 0;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    
    // Example usage
    vector<int> nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "Maximum subarray sum: " << solution.maxSubArray_method1(nums) << endl;  // Expected output: 6
    cout << "Maximum subarray sum: " << solution.maxSubArray_method2(nums) << endl;  // Expected output: 6
    return 0;
}

// g++ -std=c++17 Leetcode_0053_Maximum_Subarray.cpp -o test 