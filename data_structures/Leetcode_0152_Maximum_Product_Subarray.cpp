#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int res = nums[0];  // Initialize result to the first element
        int curMin = 1, curMax = 1;

        for (int n : nums) {
            // Calculate the product of current value with curMax and curMin
            int maxval = curMax * n;
            int minval = curMin * n;

            // Update curMax and curMin
            curMax = max({maxval, minval, n});  // Take the maximum of the 3
            curMin = min({maxval, minval, n});  // Take the minimum of the 3

            // Update result with the maximum product found so far
            res = max(res, curMax);
        }

        return res;
    }
};

int main() {
    Solution sol;

    vector<int> nums1 = {2, 3, -2, 4};
    cout << sol.maxProduct(nums1) << endl;  // Expected output: 6

    vector<int> nums2 = {-1, -2, -3, 4};
    cout << sol.maxProduct(nums2) << endl;  // Expected output: 24

    return 0;
}


// g++ -std=c++17 Leetcode_0152_Maximum_Product_Subarray.cpp -o test        