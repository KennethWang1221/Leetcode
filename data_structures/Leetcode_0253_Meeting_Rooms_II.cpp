#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;

class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        if (intervals.empty()) return 0;
        
        // Sort the intervals in ascending order of their start time
        sort(intervals.begin(), intervals.end());
        
        // Min-heap to store the end times of meetings
        priority_queue<int, vector<int>, greater<int>> heap;
        
        // Add the end time of the first meeting
        heap.push(intervals[0][1]);
        
        for (int i = 1; i < intervals.size(); ++i) {
            int curStart = intervals[i][0], curEnd = intervals[i][1];
            
            // If the current meeting starts after or when the earliest ending meeting ends
            if (curStart >= heap.top()) {
                heap.pop();  // We can reuse the room
            }
            
            // Add the current meeting's end time to the heap
            heap.push(curEnd);
        }
        
        // The size of the heap represents the number of rooms we need
        return heap.size();
    }
};

int main() {
    Solution solution;
    
    // Test case
    vector<vector<int>> intervals = {{5, 8}, {6, 8}};
    cout << solution.minMeetingRooms(intervals) << endl;  // Expected output: 2
    
    return 0;
}

// g++ Leetcode_0253_Meeting_Rooms_II.cpp -o test  && ./test