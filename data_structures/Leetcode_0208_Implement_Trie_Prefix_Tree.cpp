#include <iostream>
#include <vector>
#include <string>
using namespace std;

class TrieNode {
public:
    vector<TrieNode*> children;  // Array of children (26 letters)
    bool end;  // Indicates if this node marks the end of a word

    TrieNode() {
        children = vector<TrieNode*>(26, nullptr);
        end = false;
    }
};

class Trie {
public:
    TrieNode* root;

    Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the trie.
    void insert(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
        }
        curr->end = true;
    }

    // Returns if the word is in the trie.
    bool search(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                return false;
            }
            curr = curr->children[index];
        }
        return curr->end;
    }

    // Returns if there is any word in the trie that starts with the given prefix.
    bool startsWith(string prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                return false;
            }
            curr = curr->children[index];
        }
        return true;
    }
};

int main() {
    Trie trie;
    trie.insert("apple");
    cout << (trie.search("apple") ? "True" : "False") << endl;  // Expected output: True
    cout << (trie.search("app") ? "True" : "False") << endl;    // Expected output: False
    cout << (trie.startsWith("app") ? "True" : "False") << endl; // Expected output: True
    trie.insert("app");
    cout << (trie.search("app") ? "True" : "False") << endl;    // Expected output: True

    return 0;
}
// g++ -std=c++17 Leetcode_0052_N_Queens_II.cpp -o test