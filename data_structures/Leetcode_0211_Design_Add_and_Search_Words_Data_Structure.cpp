#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;  // Map of characters to TrieNode
    bool word;  // True if this node marks the end of a word

    TrieNode() : word(false) {}
};

class WordDictionary {
public:
    WordDictionary() {
        root = new TrieNode();  // Initialize root of Trie
    }

    // Adds a word to the dictionary
    void addWord(string word) {
        TrieNode* cur = root;
        for (char c : word) {
            if (cur->children.find(c) == cur->children.end()) {
                cur->children[c] = new TrieNode();
            }
            cur = cur->children[c];
        }
        cur->word = true;  // Mark the end of the word
    }

    // Searches for a word in the dictionary with support for wildcard '.'
    bool search(string word) {
        return dfs(0, word, root);
    }

private:
    TrieNode* root;

    // Helper function to perform DFS for the search
    bool dfs(int idx, string& word, TrieNode* node) {
        if (idx == word.size()) {
            return node->word;  // Return true if we've reached the end of the word and it's a valid word
        }

        char c = word[idx];

        if (c == '.') {
            // If the current character is '.', try all possible children nodes
            for (auto& child : node->children) {
                if (dfs(idx + 1, word, child.second)) {
                    return true;
                }
            }
            return false;
        } else {
            // If the character is not '.', proceed to the next node
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            return dfs(idx + 1, word, node->children[c]);
        }
    }
};

int main() {
    WordDictionary wordDictionary;

    // Test cases
    wordDictionary.addWord("bad");
    wordDictionary.addWord("dad");
    wordDictionary.addWord("mad");

    cout << wordDictionary.search("pad") << endl;  // Expected output: 0 (False)
    cout << wordDictionary.search("bad") << endl;  // Expected output: 1 (True)
    cout << wordDictionary.search(".ad") << endl;  // Expected output: 1 (True)
    cout << wordDictionary.search("b..") << endl;  // Expected output: 1 (True)

    return 0;
}


// g++ -std=c++17 Leetcode_0211_Design_Add_and_Search_Words_Data_Structure.cpp -o test