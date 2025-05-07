#include <iostream>
#include <vector>
#include <algorithm>  // For std::min
using namespace std;

class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        int res = nums[0];

        while (l <= r) {
            // If the leftmost element is smaller than the rightmost, it's the minimum element
            if (nums[l] < nums[r]) {
                res = min(res, nums[l]);
                break;  // No need to check further if the array is already sorted
            }

            int mid = l + (r - l) / 2;
            res = min(res, nums[mid]);

            // If the middle element is larger than the leftmost, the minimum is on the right half
            if (nums[mid] >= nums[l]) {
                l = mid + 1;
            } else {  // Otherwise, it's on the left half
                r = mid - 1;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {3, 4, 5, 1, 2};

    int result = solution.findMin(nums);
    cout << "The minimum value in the rotated sorted array is: " << result << endl;  // Expected output: 1

    return 0;
}

// g++ -std=c++17 Leetcode_0153_Find_Minimum_in_Rotated_Sorted_Array.cpp -o test