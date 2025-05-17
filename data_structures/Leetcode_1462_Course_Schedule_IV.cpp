#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>

using namespace std;

class Solution {
public:
    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        // Create an adjacency list for prerequisites
        unordered_map<int, vector<int>> adj;
        for (const auto& pre : prerequisites) {
            adj[pre[1]].push_back(pre[0]);  // Course pre[0] is a prerequisite for course pre[1]
        }

        unordered_map<int, unordered_set<int>> prereqMap;  // Map course -> set of indirect prerequisites

        // Helper function to perform DFS and find all prerequisites
        function<unordered_set<int>(int)> dfs = [&](int crs) -> unordered_set<int> {
            if (prereqMap.find(crs) != prereqMap.end()) {
                return prereqMap[crs];  // Return already computed prerequisites for the course
            }

            unordered_set<int> prereqs;
            for (int pre : adj[crs]) {
                unordered_set<int> preSet = dfs(pre);  // Get prerequisites for the current course
                prereqs.insert(preSet.begin(), preSet.end());  // Merge them into the current course's prerequisites
            }

            prereqs.insert(crs);  // Add the course itself as a prerequisite
            prereqMap[crs] = prereqs;  // Store the prerequisites for the current course
            return prereqs;
        };

        // Compute prerequisites for all courses
        for (int crs = 0; crs < numCourses; crs++) {
            dfs(crs);
        }

        // Process queries to check if course u is a prerequisite of course v
        vector<bool> res;
        for (const auto& query : queries) {
            int u = query[0], v = query[1];
            res.push_back(prereqMap[v].count(u) > 0);  // Check if u is in the set of prerequisites for v
        }

        return res;
    }
};

int main() {
    Solution sol;
    int numCourses = 2;
    vector<vector<int>> prerequisites = {{1, 0}};
    vector<vector<int>> queries = {{0, 1}, {1, 0}};
    
    vector<bool> res = sol.checkIfPrerequisite(numCourses, prerequisites, queries);

    for (bool result : res) {
        cout << (result ? "True" : "False") << " ";
    }
    cout << endl;

    return 0;
}



// g++ -std=c++17 Leetcode_1462_Course_Schedule_IV.cpp -o test