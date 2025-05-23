#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        // Edge case: If no intervals are given, return an empty result
        if (intervals.empty()) {
            return {};
        }

        // Sort intervals based on the starting point
        sort(intervals.begin(), intervals.end());

        vector<vector<int>> res;
        res.push_back(intervals[0]);  // Start with the first interval

        for (int i = 1; i < intervals.size(); ++i) {
            // Get the last merged interval and the current interval
            int prevStart = res.back()[0], prevEnd = res.back()[1];
            int curStart = intervals[i][0], curEnd = intervals[i][1];

            // Check if the current interval overlaps with the last merged interval
            if (curStart <= prevEnd) {
                // Merge intervals by updating the end of the last merged interval
                res.back()[0] = min(curStart, prevStart);
                res.back()[1] = max(curEnd, prevEnd);
            } else {
                // No overlap, so just add the current interval
                res.push_back({curStart, curEnd});
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    
    // Test case: merge intervals
    vector<vector<int>> intervals = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
    vector<vector<int>> result = solution.merge(intervals);
    
    // Print the result
    for (const auto& interval : result) {
        cout << "[" << interval[0] << ", " << interval[1] << "] ";
    }
    cout << endl;

    return 0;
}
// g++ Leetcode_0056_Merge_Intervals.cpp -o test  && ./test
