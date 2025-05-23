#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        int n = intervals.size();
        if (n == 0) return true;  // If no intervals, return true
        
        // Sort intervals by their start time
        sort(intervals.begin(), intervals.end());

        // Check for overlaps
        for (int i = 1; i < n; ++i) {
            int prevStart = intervals[i - 1][0], prevEnd = intervals[i - 1][1];
            int curStart = intervals[i][0], curEnd = intervals[i][1];

            // If the current meeting starts before the previous one ends, return false
            if (curStart < prevEnd) {
                return false;
            }
        }

        return true;
    }
};

int main() {
    Solution solution;
    
    // Test case
    vector<vector<int>> intervals = {{0, 30}, {5, 10}, {15, 20}};
    bool result = solution.canAttendMeetings(intervals);
    
    cout << (result ? "True" : "False") << endl;  // Expected output: False

    return 0;
}

// g++ Leetcode_0252_Meeting_Rooms.cpp -o test  && ./test