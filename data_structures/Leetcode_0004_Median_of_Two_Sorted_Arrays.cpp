#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        // Ensure that nums1 is the smaller array for efficiency
        if (nums1.size() > nums2.size()) {
            swap(nums1, nums2);
        }

        int l = 0, r = nums1.size();
        int total = nums1.size() + nums2.size();
        int half = total / 2;

        while (l <= r) {
            int mid1 = l + (r - l) / 2;  // Partition index for nums1
            int mid2 = half - mid1;       // Partition index for nums2

            // Elements around the partition in nums1 and nums2
            int left1 = (mid1 > 0) ? nums1[mid1 - 1] : INT_MIN;
            int right1 = (mid1 < nums1.size()) ? nums1[mid1] : INT_MAX;

            int left2 = (mid2 > 0) ? nums2[mid2 - 1] : INT_MIN;
            int right2 = (mid2 < nums2.size()) ? nums2[mid2] : INT_MAX;

            // Check if we have found the correct partition
            if (left1 <= right2 && left2 <= right1) {
                // If total length is odd, the median is the min of right1 and right2
                if (total % 2) {
                    return min(right1, right2);
                } else {
                    // If total length is even, the median is the average of max(left1, left2) and min(right1, right2)
                    return (max(left1, left2) + min(right1, right2)) / 2.0;
                }
            } else if (left1 > right2) {
                // Move the partition in nums1 to the left
                r = mid1 - 1;
            } else {
                // Move the partition in nums1 to the right
                l = mid1 + 1;
            }
        }

        return 0.0;  // This should never happen if the input arrays are sorted
    }
};

int main() {
    Solution solution;

    vector<int> nums1 = {1, 3};
    vector<int> nums2 = {2};
    double result = solution.findMedianSortedArrays(nums1, nums2);
    cout << "The median is: " << result << endl;  // Expected output: 2

    return 0;
}


// g++ -std=c++17 Leetcode_0004_Median_of_Two_Sorted_Arrays.cpp -o test
