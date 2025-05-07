#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;

        while (l <= r) {
            int mid = l + (r - l) / 2;

            if (nums[mid] == target) {
                return mid;  // Target found at mid index
            }

            // Left sorted portion
            if (nums[l] <= nums[mid]) {
                if (target > nums[mid] || target < nums[l]) {
                    l = mid + 1;  // Search in the right half
                } else {
                    r = mid - 1;  // Search in the left half
                }
            } else {  // Right sorted portion
                if (target < nums[mid] || target > nums[r]) {
                    r = mid - 1;  // Search in the left half
                } else {
                    l = mid + 1;  // Search in the right half
                }
            }
        }

        return -1;  // Target not found
    }
};

int main() {
    Solution solution;
    vector<int> nums = {4, 5, 6, 7, 0, 1, 2};
    int target = 0;

    int result = solution.search(nums, target);
    cout << "Target found at index: " << result << endl;  // Expected output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0033_Search_in_Rotated_Sorted_Array.cpp -o test
