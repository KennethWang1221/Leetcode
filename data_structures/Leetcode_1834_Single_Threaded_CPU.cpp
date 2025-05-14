#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> getOrder(vector<vector<int>>& tasks) {
        // Prepare the tasks with (arrival time, processing time, original index)
        vector<tuple<int, int, int>> sortedTasks;
        for (int i = 0; i < tasks.size(); ++i) {
            sortedTasks.push_back({tasks[i][0], tasks[i][1], i});
        }
        sort(sortedTasks.begin(), sortedTasks.end());

        vector<int> result;
        // Use long long to avoid integer overflow
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        int cur_task_index = 0;
        long long cur_time = get<0>(sortedTasks[0]);

        while (result.size() < tasks.size()) {
            // Add all tasks that have arrived by the current time
            while (cur_task_index < tasks.size() && get<0>(sortedTasks[cur_task_index]) <= cur_time) {
                pq.push({get<1>(sortedTasks[cur_task_index]), get<2>(sortedTasks[cur_task_index])});
                cur_task_index++;
            }

            if (!pq.empty()) {
                // Process the task with the shortest processing time
                auto [time_difference, original_index] = pq.top();
                pq.pop();
                cur_time += time_difference;
                result.push_back(original_index);
            } else if (cur_task_index < tasks.size()) {
                // No tasks are available, move the current time forward to the next task's arrival time
                cur_time = get<0>(sortedTasks[cur_task_index]);
            }
        }

        return result;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> tasks = {{1, 2}, {2, 4}, {3, 2}, {4, 1}};
    vector<int> order = solution.getOrder(tasks);

    // Output the result
    for (int task : order) {
        cout << task << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_1834_Single_Threaded_CPU.cpp -o test