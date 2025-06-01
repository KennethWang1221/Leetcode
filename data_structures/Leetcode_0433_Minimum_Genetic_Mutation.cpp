#include <iostream>
#include <queue>
#include <unordered_set>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    int minMutation(string startGene, string endGene, vector<string>& bank) {
        unordered_set<string> bankSet(bank.begin(), bank.end());
        if (bankSet.find(endGene) == bankSet.end()) 
            return -1;
        
        queue<pair<string, int>> q;
        q.push({startGene, 0});
        
        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            string word = current.first;
            int steps = current.second;
            
            if (word == endGene) 
                return steps;
            
            for (int i = 0; i < word.size(); i++) {
                char original_char = word[i];
                for (char ch : {'A', 'C', 'G', 'T'}) {
                    word[i] = ch;
                    if (bankSet.find(word) != bankSet.end()) {
                        bankSet.erase(word);
                        q.push({word, steps + 1});
                    }
                }
                word[i] = original_char;
            }
        }
        return -1;
    }
};

int main() {
    string startGene = "AACCGGTT";
    string endGene = "AAACGGTA";
    vector<string> bank = {"AACCGGTA", "AACCGCTA", "AAACGGTA"};
    Solution sol;
    int result = sol.minMutation(startGene, endGene, bank);
    cout << "Result: " << result << endl; // Expected output: 2
    return 0;
}
//  g++ -std=c++17 Leetcode_0433_Minimum_Genetic_Mutation.cpp -o test