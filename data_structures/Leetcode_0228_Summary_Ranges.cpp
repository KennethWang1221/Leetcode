#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> result;
        if (nums.empty()) return result;

        int start = nums[0];

        for (int i = 1; i < nums.size(); ++i) {
            // If current number breaks the sequence
            if (nums[i] != nums[i - 1] + 1) {
                // Add previous range
                if (start == nums[i - 1]) {
                    result.push_back(to_string(start));
                } else {
                    result.push_back(to_string(start) + "->" + to_string(nums[i - 1]));
                }
                start = nums[i];
            }
        }

        // Add the last range
        if (start == nums.back()) {
            result.push_back(to_string(start));
        } else {
            result.push_back(to_string(start) + "->" + to_string(nums.back()));
        }

        return result;
    }
};

// Test Case
int main() {
    Solution sol;

    vector<int> nums1 = {0, 1, 2, 4, 5, 7};
    vector<int> nums2 = {0, 2, 3, 4, 6, 8, 9};

    auto print = [](const vector<string>& res) {
        cout << "[";
        for (int i = 0; i < res.size(); ++i) {
            cout << "\"" << res[i] << "\"";
            if (i != res.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    };

    vector<string> res1 = sol.summaryRanges(nums1);
    print(res1); // Output: ["0->2", "4->5", "7"]

    vector<string> res2 = sol.summaryRanges(nums2);
    print(res2); // Output: ["0", "2->4", "6", "8->9"]

    return 0;
}

// g++ -std=c++17 Leetcode_0228_Summary_Ranges.cpp -o test