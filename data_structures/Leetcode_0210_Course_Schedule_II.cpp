#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
public:
    // Helper DFS function to check for cycle and to add the course to the result
    bool dfs(int course, unordered_map<int, vector<int>>& preMaps, unordered_set<int>& visited, unordered_set<int>& cycle, vector<int>& res) {
        // If the course is in the current cycle, it means there's a cycle
        if (cycle.count(course)) {
            return false;
        }

        // If the course is already visited, it's already processed
        if (visited.count(course)) {
            return true;
        }

        // Mark the course as being processed (part of the current DFS path)
        cycle.insert(course);

        // Explore all prerequisites for the current course
        for (int pre : preMaps[course]) {
            if (!dfs(pre, preMaps, visited, cycle, res)) {
                return false;  // If we detect a cycle, return false
            }
        }

        // Remove the course from the current cycle (it's fully processed)
        cycle.erase(course);
        visited.insert(course);
        
        // Add the course to the result in the order of processing
        res.push_back(course);

        return true;
    }

    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int, vector<int>> preMaps;  // Adjacency list to store the prerequisites
        unordered_set<int> visited;  // Set to track visited nodes (courses)
        unordered_set<int> cycle;  // Set to track courses in the current DFS path (cycle detection)
        vector<int> res;  // Result vector to store the topological order

        // Build the adjacency list (graph)
        for (const auto& pre : prerequisites) {
            preMaps[pre[0]].push_back(pre[1]);
        }

        // Check each course
        for (int course = 0; course < numCourses; course++) {
            if (!visited.count(course)) {  // If the course is not visited yet
                if (!dfs(course, preMaps, visited, cycle, res)) {
                    return {};  // Return empty list if cycle detected
                }
            }
        }

        return res;  // Return the topological order (no need to reverse)
    }
};

int main() {
    Solution sol;
    int numCourses = 2;
    vector<vector<int>> prerequisites = {{1, 0}};
    
    vector<int> res = sol.findOrder(numCourses, prerequisites);
    
    if (res.empty()) {
        cout << "It's not possible to finish all courses due to a cycle." << endl;
    } else {
        cout << "The order of courses to take is: ";
        for (int course : res) {
            cout << course << " ";
        }
        cout << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0210_Course_Schedule_II.cpp -o test