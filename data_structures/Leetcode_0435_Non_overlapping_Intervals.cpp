#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if (intervals.empty()) {
            return 0;
        }

        // Sort intervals based on the starting points
        sort(intervals.begin(), intervals.end());

        int n = intervals.size();
        int overlap = 0;
        vector<int> prev = intervals[0];

        for (int i = 1; i < n; ++i) {
            int curStart = intervals[i][0];
            int curEnd = intervals[i][1];
            int prevStart = prev[0];
            int prevEnd = prev[1];

            // If the current interval overlaps with the previous one
            if (curStart < prevEnd) {
                overlap++;
                // We select the interval with the smallest end point to minimize overlap
                prev[1] = min(curEnd, prevEnd);
            } else {
                prev = intervals[i]; // No overlap, update the previous interval to the current one
            }
        }

        return overlap;
    }
};

int main() {
    Solution solution;

    // Test case
    vector<vector<int>> intervals = {{-52, 31}, {-73, -26}, {82, 97}, {-65, -11}, 
                                     {-62, -49}, {95, 99}, {58, 95}, {-31, 49}, 
                                     {66, 98}, {-63, 2}, {30, 47}, {-40, -26}};
    
    int result = solution.eraseOverlapIntervals(intervals);
    cout << "Number of intervals to remove: " << result << endl;  // Expected output based on the input
    
    return 0;
}

// g++ Leetcode_0435_Non_overlapping_Intervals.cpp -o test  && ./test