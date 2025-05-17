#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // Create the adjacency list for the course graph
        unordered_map<int, vector<int>> preMaps;
        unordered_set<int> visited;

        // Initialize the adjacency list
        for (auto& pre : prerequisites) {
            preMaps[pre[0]].push_back(pre[1]);
        }

        // Helper DFS function to check for cycles
        function<bool(int)> dfs = [&](int c) {
            if (preMaps[c].empty()) {
                return true;  // No prerequisites, no cycle
            }
            if (visited.count(c)) {
                return false;  // Cycle detected
            }
            
            visited.insert(c);  // Mark the current course as visited
            for (int pre : preMaps[c]) {
                if (!dfs(pre)) {
                    return false;  // Cycle detected in the prerequisites
                }
            }
            
            preMaps[c].clear();  // Mark this course as processed
            visited.erase(c);  // Remove from visited set
            return true;
        };

        // Check each course to see if it can be completed
        for (int c = 0; c < numCourses; c++) {
            if (!dfs(c)) {
                return false;  // If any course can't be completed, return false
            }
        }

        return true;  // All courses can be completed
    }
};

int main() {
    Solution sol;
    int numCourses = 5;
    vector<vector<int>> prerequisites = {{1, 4}, {2, 4}, {3, 1}, {3, 2}};
    
    bool res = sol.canFinish(numCourses, prerequisites);
    cout << "Can finish all courses: " << (res ? "Yes" : "No") << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0207_Course_Schedule.cpp -o test