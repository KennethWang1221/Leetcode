#include <iostream>
#include <vector>
#include <deque>
using namespace std;
// Monotonic Queue
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> output;
        deque<int> q;  // stores indices of elements in the current window
        int n = nums.size();
        
        for (int r = 0; r < n; r++) {
            // Pop smaller values from the back of the deque
            while (!q.empty() && nums[q.back()] < nums[r]) {
                q.pop_back();
            }
            q.push_back(r);

            // Remove the left-most value if it's outside the window
            if (q.front() < r - k + 1) {
                q.pop_front();
            }

            // Add the largest element of the window to the output
            if (r >= k - 1) {
                output.push_back(nums[q.front()]);
            }
        }

        return output;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 3, -1, -3, 5, 3, 6, 7};
    int k = 3;

    vector<int> result = solution.maxSlidingWindow(nums, k);

    cout << "Sliding window maximums: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0239_Sliding_Window_Maximum.cpp -o test