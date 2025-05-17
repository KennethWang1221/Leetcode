#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <queue>

using namespace std;

class UnionFind {
public:
    UnionFind(int n) {
        par.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; ++i) {
            par[i] = i;
        }
    }

    int find(int x) {
        while (x != par[x]) {
            par[x] = par[par[x]];  // Path compression
            x = par[x];
        }
        return x;
    }

    bool unionSets(int x1, int x2) {
        int p1 = find(x1), p2 = find(x2);
        if (p1 == p2) return false;

        // Union by rank
        if (rank[p1] > rank[p2]) {
            par[p2] = p1;
            rank[p1] += rank[p2];
        } else {
            par[p1] = p2;
            rank[p2] += rank[p1];
        }
        return true;
    }

private:
    vector<int> par;
    vector<int> rank;
};

class Solution {
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        int n = accounts.size();
        UnionFind uf(n);

        unordered_map<string, int> emailToAcc; // email -> account index

        // Step 1: Union accounts based on common emails
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j < accounts[i].size(); ++j) {
                string email = accounts[i][j];
                if (emailToAcc.find(email) != emailToAcc.end()) {
                    uf.unionSets(i, emailToAcc[email]);  // Union account i and account that has this email
                } else {
                    emailToAcc[email] = i;  // Map email to current account index
                }
            }
        }

        // Step 2: Group emails by their root account index
        unordered_map<int, unordered_set<string>> emailGroup; // account index -> set of emails
        for (auto& entry : emailToAcc) {
            int leader = uf.find(entry.second);  // Get the root of the account
            emailGroup[leader].insert(entry.first); // Group emails by the root
        }

        // Step 3: Prepare the result
        vector<vector<string>> res;
        for (auto& entry : emailGroup) {
            int idx = entry.first;
            vector<string> emails(entry.second.begin(), entry.second.end());
            sort(emails.begin(), emails.end()); // Sort emails alphabetically
            emails.insert(emails.begin(), accounts[idx][0]); // Add the account name
            res.push_back(emails);
        }

        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<string>> accounts = {
        {"John", "johnsmith@mail.com", "john00@mail.com"},
        {"John", "johnnybravo@mail.com"},
        {"John", "johnsmith@mail.com", "john_newyork@mail.com"},
        {"Mary", "mary@mail.com"}
    };

    vector<vector<string>> mergedAccounts = sol.accountsMerge(accounts);
    
    for (const auto& account : mergedAccounts) {
        for (const string& email : account) {
            cout << email << " ";
        }
        cout << endl;
    }

    return 0;
}
// g++ -std=c++17 Leetcode_0721_Accounts_Merge.cpp -o test