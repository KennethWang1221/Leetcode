#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
private:
    bool dfs(int node, vector<vector<int>>& graph, vector<int>& visited, vector<int>& order, vector<int>& path) {
        visited[node] = 1;  // visiting
        path[node] = 1;
        
        for (int nei : graph[node]) {
            if (visited[nei] == 0) {
                if (!dfs(nei, graph, visited, order, path))
                    return false;
            } else if (path[nei]) {
                return false;  // cycle detected
            }
        }
        
        path[node] = 0;
        order.push_back(node);
        visited[node] = 2;  // visited
        return true;
    }
    
    vector<int> topologicalSort(int k, vector<vector<int>>& conditions) {
        vector<vector<int>> graph(k+1);
        for (auto& cond : conditions) {
            int u = cond[0], v = cond[1];
            graph[u].push_back(v);
        }
        
        vector<int> visited(k+1, 0);  // 0 = unvisited, 1 = visiting, 2 = visited
        vector<int> order;
        vector<int> path(k+1, 0);
        
        for (int i = 1; i <= k; ++i) {
            if (visited[i] == 0) {
                if (!dfs(i, graph, visited, order, path))
                    return {};  // cycle detected
            }
        }
        
        reverse(order.begin(), order.end());  // We need to reverse since we add at end of DFS
        return order;
    }

public:
    vector<vector<int>> buildMatrix(int k, vector<vector<int>>& rowConditions, vector<vector<int>>& colConditions) {
        vector<int> rowOrder = topologicalSort(k, rowConditions);
        vector<int> colOrder = topologicalSort(k, colConditions);
        
        if (rowOrder.empty() || colOrder.empty())
            return {};  // Not possible to construct the matrix
        
        vector<vector<int>> result(k, vector<int>(k, 0));
        vector<int> pos(k+1, 0);  // value -> column position
        
        for (int j = 0; j < k; ++j)
            pos[colOrder[j]] = j;
        
        for (int i = 0; i < k; ++i)
            result[i][pos[rowOrder[i]]] = rowOrder[i];
        
        return result;
    }
};

int main() {
    Solution sol;

    int k = 3;
    vector<vector<int>> rowConditions = {{1, 2}, {3, 2}};
    vector<vector<int>> colConditions = {{2, 1}, {3, 2}};
    
    vector<vector<int>> result = sol.buildMatrix(k, rowConditions, colConditions);

    if (result.empty()) {
        cout << "No valid matrix can be built." << endl;
    } else {
        for (const auto& row : result) {
            for (int value : row) {
                cout << value << " ";
            }
            cout << endl;
        }
    }

    return 0;
}
// g++ -std=c++17 Leetcode_2392_Build_a_Matrix_With_Conditions.cpp -o test