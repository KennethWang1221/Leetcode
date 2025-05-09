#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <deque>
using namespace std;

class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        unordered_map<char, int> counts;  // Map to count the frequency of each task

        // Count the frequency of each task
        for (char task : tasks) {
            counts[task]++;
        }

        priority_queue<int> maxHeap;  // Max-heap to store the task frequencies

        // Push the negative of task frequencies to make a max-heap
        for (auto& entry : counts) {
            maxHeap.push(entry.second);
        }

        int time = 0;
        deque<pair<int, int>> q;  // Deque to store tasks in cooldown with their available time

        while (!maxHeap.empty() || !q.empty()) {
            time++;

            if (!maxHeap.empty()) {
                // Pop the task with the highest frequency (most frequent task)
                int cnt = maxHeap.top();
                maxHeap.pop();
                cnt--;  // Decrement the count of the current task

                if (cnt > 0) {
                    // If there are more occurrences of the task, push it to the cooldown queue
                    q.push_back({cnt, time + n});
                }
            } else {
                // If no tasks are ready to be processed, jump time forward to when the first task becomes available
                time = q.front().second;
            }

            // Check if any task in cooldown is now available
            if (!q.empty() && q.front().second == time) {
                maxHeap.push(q.front().first);
                q.pop_front();
            }
        }

        return time;
    }
};

int main() {
    Solution solution;
    vector<char> tasks = {'A', 'A', 'A', 'B', 'B', 'B'};
    int n = 2;
    cout << solution.leastInterval(tasks, n) << endl;  // Output should be 8

    return 0;
}


// g++ -std=c++17 Leetcode_0621_Task_Scheduler.cpp -o test