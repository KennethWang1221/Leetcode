#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int cursum = 0;
        int res = 0;
        unordered_map<int, int> prefixsum;
        prefixsum[0] = 1;  // Base case: there is one way to get sum 0 (empty subarray)

        for (int n : nums) {
            cursum += n;  // Update the current sum

            // Check if there is a subarray with sum = cursum - k
            int diff = cursum - k;
            if (prefixsum.find(diff) != prefixsum.end()) {
                res += prefixsum[diff];  // Add the number of times diff has been seen
            }

            // Update the prefix sum map
            prefixsum[cursum]++;
        }

        return res;
    }
};

int main() {
    Solution solution;

    vector<int> nums = {1, 1, 1};
    int k = 2;
    int res = solution.subarraySum(nums, k);
    cout << res << endl;  // Expected output: 2 (subarrays [1, 1] and [1, 1])

    nums = {1, 2, 3};
    k = 3;
    res = solution.subarraySum(nums, k);
    cout << res << endl;  // Expected output: 2 (subarrays [1, 2] and [3])

    return 0;
}


// g++ -std=c++17 Leetcode_0560_Subarray_Sum_Equals_K.cpp -o test