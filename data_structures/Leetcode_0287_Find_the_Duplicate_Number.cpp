#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        // Floyd's Tortoise and Hare (Cycle Detection)
        int slow = 0, fast = 0;

        // Phase 1: Finding the intersection point in the cycle
        do {
            slow = nums[slow];             // Move slow pointer by 1 step
            fast = nums[nums[fast]];       // Move fast pointer by 2 steps
        } while (slow != fast);

        // Phase 2: Finding the entrance to the cycle (duplicate number)
        int slow2 = 0;
        while (slow != slow2) {
            slow = nums[slow];            // Move slow pointer by 1 step
            slow2 = nums[slow2];          // Move second slow pointer by 1 step
        }

        return slow; // The duplicate number
    }
};

// Function to print the result
void test_case(vector<int>& nums) {
    Solution s;
    int result = s.findDuplicate(nums);
    cout << "The duplicate number is: " << result << endl;
}

int main() {
    // Test case 1
    vector<int> nums1 = {1, 3, 4, 2, 2};
    test_case(nums1);

    // Test case 2
    vector<int> nums2 = {3, 1, 3, 4, 2};
    test_case(nums2);

    // Test case 3
    vector<int> nums3 = {5, 4, 6, 7, 2, 3, 5, 9};
    test_case(nums3);

    return 0;
}
// g++ -std=c++17 Leetcode_0287_Find_the_Duplicate_Number.cpp -o test
