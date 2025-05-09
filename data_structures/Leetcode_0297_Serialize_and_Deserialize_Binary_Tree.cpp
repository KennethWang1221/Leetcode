#include <iostream>
#include <sstream>
#include <queue>
#include <string>
#include <vector>

using namespace std;

// Definition for a binary tree node.
class TreeNode {
public:
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Codec {
public:
    // Serialize a tree to a string
    string serialize(TreeNode* root) {
        stringstream ss;
        dfsSerialize(root, ss);
        return ss.str();
    }

    // Deserialize a string back to a tree
    TreeNode* deserialize(string data) {
        stringstream ss(data);
        return dfsDeserialize(ss);
    }

private:
    // Helper function for serialization
    void dfsSerialize(TreeNode* node, stringstream& ss) {
        if (!node) {
            ss << "N,";
            return;
        }
        ss << node->val << ",";
        dfsSerialize(node->left, ss);
        dfsSerialize(node->right, ss);
    }

    // Helper function for deserialization
    TreeNode* dfsDeserialize(stringstream& ss) {
        string val;
        getline(ss, val, ',');
        
        if (val == "N") {
            return nullptr;
        }

        TreeNode* node = new TreeNode(stoi(val));
        node->left = dfsDeserialize(ss);
        node->right = dfsDeserialize(ss);
        return node;
    }
};

int main() {
    // Test case 1: Create a tree and test serialization/deserialization
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);

    Codec codec;

    // Serialize the tree
    string serialized = codec.serialize(root);
    cout << "Serialized Tree: " << serialized << endl;

    // Deserialize the tree
    TreeNode* deserialized = codec.deserialize(serialized);
    cout << "Deserialized Tree (root value): " << deserialized->val << endl;

    // Test case 2: Serialize and Deserialize a tree with None nodes
    TreeNode* root2 = new TreeNode(1);
    root2->left = nullptr;
    root2->right = new TreeNode(3);

    string serialized2 = codec.serialize(root2);
    cout << "Serialized Tree 2: " << serialized2 << endl;

    TreeNode* deserialized2 = codec.deserialize(serialized2);
    cout << "Deserialized Tree 2 (root value): " << deserialized2->val << endl;
    cout << "Deserialized Tree 2 (right child value): " << deserialized2->right->val << endl;

    // Clean up dynamically allocated memory
    delete root;
    delete root2;
    delete deserialized;
    delete deserialized2;

    return 0;
}



// g++ -std=c++17 Leetcode_0297_Serialize_and_Deserialize_Binary_Tree.cpp -o test