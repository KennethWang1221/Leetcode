#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

using namespace std;

class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        // If the endWord is not in the wordList, return 0
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        if (wordSet.find(endWord) == wordSet.end()) {
            return 0;
        }

        // Build the adjacency list for words (each word can be transformed to others)
        unordered_map<string, vector<string>> nei;
        wordSet.insert(beginWord);  // Include the beginWord in the set
        for (const string& word : wordSet) {
            for (int j = 0; j < word.size(); j++) {
                string pattern = word.substr(0, j) + "*" + word.substr(j + 1);
                nei[pattern].push_back(word);
            }
        }

        // BFS to find the shortest path
        queue<string> q;
        unordered_set<string> visited;
        q.push(beginWord);
        visited.insert(beginWord);
        int res = 1;

        while (!q.empty()) {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; ++i) {
                string word = q.front();
                q.pop();
                if (word == endWord) {
                    return res;  // If we reach the endWord, return the length of the path
                }

                // Explore all possible transformations of the current word
                for (int j = 0; j < word.size(); ++j) {
                    string pattern = word.substr(0, j) + "*" + word.substr(j + 1);
                    for (const string& neighbor : nei[pattern]) {
                        if (visited.find(neighbor) == visited.end()) {
                            visited.insert(neighbor);
                            q.push(neighbor);
                        }
                    }
                }
            }
            res++;
        }

        return 0;  // If no transformation sequence found
    }
};

int main() {
    Solution sol;
    vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
    string beginWord = "hit";
    string endWord = "cog";
    
    int result = sol.ladderLength(beginWord, endWord, wordList);
    cout << "Ladder Length: " << result << endl;
    
    return 0;
}
// g++ -std=c++17 Leetcode_0127_Word_Ladder.cpp -o test 