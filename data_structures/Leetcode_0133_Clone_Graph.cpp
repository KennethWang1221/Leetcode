#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <functional>
using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;

    Node() {}

    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

class Solution {
public:
    // DFS Approach
    Node* cloneGraphDFS(Node* node) {
        if (!node) return nullptr;
        
        // HashMap to store the already visited nodes
        unordered_map<Node*, Node*> visited;
        
        function<Node*(Node*)> dfs = [&](Node* node) {
            if (visited.count(node)) {
                return visited[node];  // Return the cloned node if already visited
            }
            
            // Clone the current node
            Node* clone = new Node(node->val, {});
            visited[node] = clone;  // Store the cloned node in the map

            // Recursively clone all the neighbors
            for (auto& neighbor : node->neighbors) {
                clone->neighbors.push_back(dfs(neighbor));
            }
            return clone;
        };

        return dfs(node);
    }

    // BFS Approach
    Node* cloneGraphBFS(Node* node) {
        if (!node) return nullptr;
        
        // HashMap to store the already visited nodes
        unordered_map<Node*, Node*> visited;
        queue<Node*> q;
        
        // Clone the starting node
        Node* clone = new Node(node->val, {});
        visited[node] = clone;
        q.push(node);
        
        while (!q.empty()) {
            Node* current = q.front();
            q.pop();
            
            // For each neighbor, either clone it or use the existing one
            for (auto& neighbor : current->neighbors) {
                if (visited.count(neighbor) == 0) {
                    // If not visited, clone and add to the queue
                    Node* neighborClone = new Node(neighbor->val, {});
                    visited[neighbor] = neighborClone;
                    q.push(neighbor);
                }
                // Add the cloned neighbor to the current node's neighbors
                visited[current]->neighbors.push_back(visited[neighbor]);
            }
        }

        return visited[node];  // Return the cloned starting node
    }
};

int main() {
    Solution solution;

    // Create a simple graph with nodes
    Node* node1 = new Node(1, {});
    Node* node2 = new Node(2, {});
    Node* node3 = new Node(3, {});
    Node* node4 = new Node(4, {});

    node1->neighbors = {node2, node4};
    node2->neighbors = {node1, node3};
    node3->neighbors = {node2, node4};
    node4->neighbors = {node1, node3};

    // Clone the graph using DFS
    Node* clonedGraphDFS = solution.cloneGraphDFS(node1);

    // Clone the graph using BFS
    Node* clonedGraphBFS = solution.cloneGraphBFS(node1);

    // Both clonedGraphDFS and clonedGraphBFS should now be deep copies of the original graph.

    return 0;
}


// g++ -std=c++17 Leetcode_0133_Clone_Graph.cpp -o test
