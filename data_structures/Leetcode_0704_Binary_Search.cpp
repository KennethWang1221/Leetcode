#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2; // Prevents potential overflow
            if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    int search_method2(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0;
        int r = n - 1;
        while (l <= r){
            int middle = (l+r) / 2;
            if (target > nums[middle]){
                l = middle + 1;
            } else if (target < nums[middle]){
                r = middle - 1;
            } else {
                return middle;
            }
        }
        return -1;
    }
};

// Example usage
int main() {
    Solution sol;
    vector<int> nums = {-1, 0, 3, 5, 9, 12};
    int target = 9;
    cout << sol.search(nums, target) << endl; // Output: 4
    target = 2;
    cout << sol.search(nums, target) << endl; // Output: -1
    cout << sol.search_method2(nums, target) << endl; // Output: -1
    return 0;
}


// g++ -std=c++17 Leetcode_0704_Binary_Search.cpp -o test