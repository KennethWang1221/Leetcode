#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxSubarraySumCircular(vector<int>& nums) {
        int globMax = nums[0], globMin = nums[0];
        int curMax = 0, curMin = 0;
        int total = 0;

        // Traverse the array and calculate the maximum and minimum subarray sums
        for (int n : nums) {
            curMax = max(curMax + n, n);
            curMin = min(curMin + n, n);
            total += n;
            globMax = max(curMax, globMax);
            globMin = min(curMin, globMin);
        }

        // If the maximum subarray sum is greater than 0, we can compute the circular case
        int result = (globMax > 0) ? max(globMax, total - globMin) : globMax;
        return result;
    }
};

int main() {
    Solution solution;

    // Test cases
    vector<int> nums1 = {1, -2, 3, -2};
    vector<int> nums2 = {5, -3, 5};
    vector<int> nums3 = {-5, -3, -5};

    cout << "Max circular subarray sum (Test case 1): " << solution.maxSubarraySumCircular(nums1) << endl;  // Expected output: 3
    cout << "Max circular subarray sum (Test case 2): " << solution.maxSubarraySumCircular(nums2) << endl;  // Expected output: 10
    cout << "Max circular subarray sum (Test case 3): " << solution.maxSubarraySumCircular(nums3) << endl;  // Expected output: -3

    return 0;
}

// g++ -std=c++17 Leetcode_0918_Maximum_Sum_Circular_Subarray.cpp -o test 