#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    // Helper function to perform backtracking and build the itinerary
    void backtracking(unordered_map<string, vector<string>>& adj, string src, vector<string>& res) {
        if (adj.find(src) != adj.end()) {
            vector<string>& destinations = adj[src];
            while (!destinations.empty()) {
                string dest = destinations[0];
                destinations.erase(destinations.begin());  // Remove the first destination
                backtracking(adj, dest, res);
            }
        }
        res.push_back(src);  // Add the current source to the itinerary
    }

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        unordered_map<string, vector<string>> adj;

        // Build the adjacency list
        for (const auto& ticket : tickets) {
            adj[ticket[0]].push_back(ticket[1]);
        }

        // Sort each adjacency list (destinations) to make sure we follow lexical order
        for (auto& entry : adj) {
            sort(entry.second.begin(), entry.second.end());
        }

        vector<string> res;
        backtracking(adj, "JFK", res);

        reverse(res.begin(), res.end());  // Reverse the result to get the correct order
        
        // Check if the number of visited places equals the number of tickets + 1
        if (res.size() != tickets.size() + 1) {
            return {};  // If not, return an empty vector
        }

        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<string>> tickets = {
        {"JFK", "SFO"}, {"JFK", "ATL"}, {"SFO", "ATL"}, {"ATL", "JFK"}, {"ATL", "SFO"}
    };

    vector<string> itinerary = sol.findItinerary(tickets);

    cout << "Itinerary: ";
    for (const string& airport : itinerary) {
        cout << airport << " ";
    }
    cout << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0743_Network_Delay_Time.cpp -o test 