#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    // Helper function to check if we can split the array with the largest sum as `largest`
    bool canSplit(const vector<int>& nums, int largest, int k) {
        int subarray = 0;
        int curSum = 0;
        for (int n : nums) {
            curSum += n;
            if (curSum > largest) {
                subarray++;
                curSum = n;
            }
        }
        return subarray + 1 <= k;
    }

    int splitArray(vector<int>& nums, int k) {
        int l = *max_element(nums.begin(), nums.end());  // the smallest possible largest sum
        int r = accumulate(nums.begin(), nums.end(), 0);  // the largest possible sum (sum of all elements)
        int res = r;

        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (canSplit(nums, mid, k)) {
                res = mid;
                r = mid - 1;  // Try to minimize the largest sum
            } else {
                l = mid + 1;  // Need a larger sum, increase the lower bound
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {7, 2, 5, 10, 8};
    int k = 2;

    int result = solution.splitArray(nums, k);
    cout << "The minimum largest sum for " << k << " subarrays is: " << result << endl;  // Expected output: 18

    return 0;
}

// g++ -std=c++17 Leetcode_0410_Split_Array_Largest_Sum.cpp -o test
