#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int> maxheap;  // Max-heap to store the elements

        // Push all numbers into the max-heap
        for (int num : nums) {
            maxheap.push(num);
        }

        // Pop elements from the heap until we reach the k-th largest
        for (int i = 0; i < k - 1; ++i) {
            maxheap.pop();  // Remove the largest element
        }

        // The k-th largest element is now at the top of the heap
        return maxheap.top();
    }
};

int main() {
    Solution solution;
    vector<int> nums = {3, 2, 1, 5, 6, 4};
    int k = 2;
    int res = solution.findKthLargest(nums, k);

    cout << res << endl;  // Output should be 5

    return 0;
}


// g++ -std=c++17 Leetcode_0215_Kth_Largest_Element_in_an_Array.cpp -o test